import torch
import torch.nn as nn
import numpy as np
from torch.distributions import MultivariateNormal
import torch.nn.functional as F
################################## set device ##################################
"""
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
"""

################################## PPO Policy ##################################
class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, device, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = device


    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 1)
        
        self.l4 = nn.Linear(state_dim + action_dim, 64)
        self.l5 = nn.Linear(64, 64)
        self.l6 = nn.Linear(64, 1)
        
    def Q1(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)
    
    def Q2(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.l4(x))
        x = F.relu(self.l5(x))
        return self.l6(x)

class Var(nn.Module):
    def __init__(self, action_dim, minn, maxx):
        super(Var, self).__init__()
        self.var = nn.Parameter(torch.zeros(action_dim)) # (torch.ones(action_dim) * (maxx + minn) / 2)
        self.minn, self.maxx = minn, maxx
    def get_var(self):
        print("var:", self.var)
        return F.sigmoid(self.var) * (self.maxx - self.minn) / 2 + (self.maxx + self.minn) / 2 # [-1, 1] -> [-(maxx - minn) / 2, (maxx - minn) / 2] [minn, maxx]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, device):
        super(ActorCritic, self).__init__()
        self.device = device
        self.action_dim = action_dim
        self.action_var = Var(action_dim, -5, 2)
        # nn.Parameter(torch.full((action_dim,), action_std_init * action_std_init))
        
        # actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        ).to(device)
        
        # critic (Q-value)
        self.critic = Critic(state_dim, action_dim).to(device)
        
    #def set_action_std(self, new_action_std):
    #    self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.device)

    def forward(self):
        raise NotImplementedError
    
    def act(self, state, deterministic=False): # , unsqueeze=True
        action_mean = self.actor(state)
        
        
        #if unsqueeze: cov_mat = torch.diag(self.action_var.get_var()).unsqueeze(dim=0).to(self.device)
        #else: cov_mat = torch.diag(self.action_var.get_var()).to(self.device)
        
        cov_mat = torch.diag(self.action_var.get_var()).to(self.device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        
        if deterministic: action = action_mean
        else: action = dist.sample()
        
        action_logprob = dist.log_prob(action)
        
        #print("act - action.shape:", action.shape, "logprob shape:", action_logprob.shape)
        
        q1, q2 = self.critic.Q1(state, action), self.critic.Q2(state, action)

        return action.detach(), action_logprob.detach(), q1.detach(), q2.detach()
    
    def evaluate(self, state, action):
        action_mean = self.actor(state)
        action_var = self.action_var.get_var().expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)
        dist = MultivariateNormal(action_mean, cov_mat)
        
        # For Single Action Environments.
        if self.action_dim == 1:
            action = action.reshape(-1, self.action_dim)
        
        action_logprobs = dist.log_prob(action)
        
        #print("evaluate - action.shape:", action.shape, "logprob shape:", action_logprobs.shape)
        
        dist_entropy = dist.entropy()
        q1, q2 = self.critic.Q1(state, action), self.critic.Q2(state, action)
        # 注意到现在var好像没有grad？
        return action_logprobs, q1, q2, dist_entropy


class SAC:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, device):

        self.gamma = gamma
        
        self.buffer = ReplayBuffer(device=device, max_size=1000000, state_dim=state_dim, action_dim=action_dim)

        self.policy = ActorCritic(state_dim, action_dim, device)
        self.actor_optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor}
        ])

        self.critic_optimizer = torch.optim.Adam([
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ]) 
 
        self.policy_old = ActorCritic(state_dim, action_dim, device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        self.device = device
    """
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
    """
    def select_action(self, state):
        """
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob, q1, q2 = self.policy_old.act(state)

        return action, action_logprob, torch.min(q1, q2)
        """
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob, q1, q2 = self.policy_old.act(state, deterministic=True)

        return action.detach().cpu().numpy().flatten()
        
    def select_action_withrandom(self, state): # is stochastic distribution
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob, q1, q2 = self.policy_old.act(state)

        return action.detach().cpu().numpy().flatten()       

    def select_action_eval(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob, q1, q2 = self.policy_old.act(state, determinstic=True)

        return action.detach().cpu().numpy().flatten()

    def update(self): 
    
        ALPHA = 0.2 # temperature 
        NUM_STEPS = 4000 # same as update interval
        BATCH_SIZE = 256
        tau = 0.005 
        # Optimize policy for K epochs
        for _ in range(NUM_STEPS): # sample from buffer
        
            old_states, old_actions, old_next_states, reward, is_not_terminal = self.buffer.sample(BATCH_SIZE)
            old_states = old_states.to(self.device)
            old_actions = old_actions.to(self.device)
            old_next_states = old_next_states.to(self.device)
            reward = reward.to(self.device)
            is_not_terminal = is_not_terminal.to(self.device).int()
            
            # print('shape:', old_states.shape, old_actions.shape, old_next_states.shape, reward.shape, is_not_terminal.shape)
            # torch.Size([256, 11]) torch.Size([256, 3]) torch.Size([256, 11]) torch.Size([256, 1]) torch.Size([256, 1])
            
            # Evaluating old actions and values
            logprobs, q1, q2, dist_entropy = self.policy.evaluate(old_states, old_actions)
            _, logprobs_prime, q1_prime, q2_prime = self.policy_old.act(old_next_states)
            # print('shape:', logprobs.shape, logprobs_prime.shape)
            q_target = reward + self.gamma * is_not_terminal * (torch.min(q1_prime, q2_prime) - ALPHA * logprobs_prime.reshape(-1, 1))
            #print("q_target:", q_target.shape, q1.shape, q2.shape)
            critic_loss = (self.MseLoss(q1, q_target) + self.MseLoss(q2, q_target)) / 2
            #print("cl:", critic_loss)
            # match state_values tensor dimensions with rewards tensor
            #state_values = torch.squeeze(state_values) # this is a MC implementation, not TD
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            # Finding the ratio (pi_theta / pi_theta__old)
            #ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss  
            #surr1 = ratios * advantages
            #surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            # print((torch.min(q1, q2) - ALPHA * logprobs.reshape(-1, 1)).shape)
            
            logprobs, q1, q2, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            actor_loss = -(torch.min(q1, q2) - ALPHA * logprobs.reshape(-1, 1)).mean()
             
            # final loss of clipped objective PPO
            
            # take gradient step
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Copy new weights into old policy
            for param, target_param in zip(self.policy.critic.parameters(), self.policy_old.critic.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for param, target_param in zip(self.policy.actor.parameters(), self.policy_old.actor.parameters()):
                target_param.data.copy_(param.data)
                

    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
