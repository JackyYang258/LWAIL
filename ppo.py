import torch
import torch.nn as nn
import numpy as np
from torch.distributions import MultivariateNormal
'''
################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if torch.cuda.is_available(): 
    device = torch.device('cuda:4') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")
'''

################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self, max_buffer_size, state_dim, action_dim):
        print("RolloutBuffer initialized with max_buffer_size: ", max_buffer_size)
        self.max_buffer_size = max_buffer_size
        self.ptr = 0
        # modified on 2025/11/20
        '''
        # 预分配张量空间
        self.state = torch.zeros((max_buffer_size, state_dim), dtype=torch.float32).to(device)
        self.action = torch.zeros((max_buffer_size, action_dim), dtype=torch.float32).to(device)
        self.logprob = torch.zeros(max_buffer_size, dtype=torch.float32).to(device)
        self.reward = torch.zeros(max_buffer_size, dtype=torch.float32).to(device)
        self.state_value = torch.zeros(max_buffer_size, dtype=torch.float32).to(device)
        self.is_terminal = torch.zeros(max_buffer_size, dtype=torch.float32).to(device)
        self.next_state = torch.zeros((max_buffer_size, state_dim), dtype=torch.float32).to(device)
        '''
        self.state = np.zeros((max_buffer_size, state_dim)).astype('float32')
        self.action = np.zeros((max_buffer_size, action_dim)).astype('float32')
        self.logprob = np.zeros(max_buffer_size).astype('float32')
        self.reward = np.zeros(max_buffer_size).astype('float32')
        self.state_value = np.zeros(max_buffer_size).astype('float32')
        self.is_terminal = np.zeros(max_buffer_size).astype('float32')
        self.next_state = np.zeros((max_buffer_size, state_dim)).astype('float32')
        
    def store(self, state, action, logprob, state_value, next_state, reward, is_terminal):
        if self.ptr < self.max_buffer_size:
            # 存储数据到预分配的张量中
            self.state[self.ptr] = state
            self.action[self.ptr] = action
            self.logprob[self.ptr] = logprob
            self.state_value[self.ptr] = state_value
            self.next_state[self.ptr] = next_state
            self.reward[self.ptr] = reward
            self.is_terminal[self.ptr] = is_terminal
            self.ptr += 1
        else:
            print("Buffer overflow! Consider increasing buffer size.")
    
    def clear(self):
        assert self.ptr == self.max_buffer_size
        self.ptr = 0


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init, device):
        super(ActorCritic, self).__init__()
        self.device = device
        self.action_dim = action_dim
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        
        # actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        ).to(self.device)
        
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(self.device)
        
    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)

    def forward(self):
        raise NotImplementedError
    
    def act(self, state):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, state, action):
        action_mean = self.actor(state)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)
        dist = MultivariateNormal(action_mean, cov_mat)
        
        # For Single Action Environments.
        if self.action_dim == 1:
            action = action.reshape(-1, self.action_dim)
        
        action_logprobs = dist.log_prob(action)
        
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, args, device, K_epochs=10, batch_size=64, eps_clip=0.2,  action_std_init=0.6):
        self.action_std = action_std_init
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.batch_size = batch_size # added 2025/11/20
        self.device = device
        
        self.buffer = RolloutBuffer(max_buffer_size=args.update_timestep, state_dim=state_dim, action_dim=action_dim)

        self.policy = ActorCritic(state_dim, action_dim, action_std_init, device=self.device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim, action_std_init, device=self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        

    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if self.action_std <= min_action_std:
            self.action_std = min_action_std
        self.set_action_std(self.action_std)

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            # print("device:", self.device)
            action, action_logprob, state_val = self.policy_old.act(state)

        return action, action_logprob, state_val
    
    def select_action_eval(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob, state_val = self.policy_old.act(state)

        return action.detach().cpu().numpy().flatten()

    def update(self):
        # modified on 2025/11/20
        #reward = torch.squeeze(self.buffer.reward).to(self.device)
        #is_terminal = torch.squeeze(self.buffer.is_terminal).to(self.device)
        reward = torch.from_numpy(self.buffer.reward).to(self.device)
        is_terminal = torch.from_numpy(self.buffer.is_terminal).to(self.device)
        # Monte Carlo estimate of returns
        rewards = torch.zeros_like(reward).to(self.device)
        discounted_reward = 0

        for t in reversed(range(len(reward))):
            if is_terminal[t]:
                discounted_reward = 0
            discounted_reward = reward[t] + (self.gamma * discounted_reward)
            rewards[t] = discounted_reward

        # Normalizing the rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.from_numpy(self.buffer.state).to(self.device)
        old_actions = torch.from_numpy(self.buffer.action).to(self.device)
        old_logprobs = torch.from_numpy(self.buffer.logprob).to(self.device)
        old_state_values = torch.from_numpy(self.buffer.state_value).to(self.device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()
        # modified on 2025/11/20
        for _ in range(self.K_epochs):
            N = old_states.shape[0]
            idx = torch.randperm(N)
            for i in range((N - 1) // self.batch_size + 1):
                old_states_batch, old_actions_batch, old_logprobs_batch = old_states[i*self.batch_size:(i+1)*self.batch_size], old_actions[i*self.batch_size:(i+1)*self.batch_size], old_logprobs[i*self.batch_size:(i+1)*self.batch_size]
                advantage_batch, rewards_batch = advantages[i*self.batch_size:(i+1)*self.batch_size], rewards[i*self.batch_size:(i+1)*self.batch_size]
                # Evaluating old actions and values
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states_batch, old_actions_batch)
    
                # match state_values tensor dimensions with rewards tensor 
                state_values = torch.squeeze(state_values)
                
                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs - old_logprobs_batch.detach())
    
                # Finding Surrogate Loss  
                surr1 = ratios * advantage_batch
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantage_batch
    
                # final loss of clipped objective PPO
                loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards_batch) - 0.01 * dist_entropy
                
                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
        '''
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor 
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        ''' 
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
