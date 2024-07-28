import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F
from network import network_weight_matrices

def train(expert_buffer, sample_buffer, f_net, actor_net, critic_net, phi, f_net_optimizer, m=10):
    # Draw trajectory
    s1, s1_prime = expert_buffer['states'], expert_buffer['next_states']                                    
    s2, s2_prime = sample_buffer['states'], sample_buffer['next_states']
        
    for iter_f in range(m):
        # Calculate the loss
        loss_f = (torch.mean(f_net(phi(s2), phi(s2_prime))) - 
                  torch.mean(f_net(phi(s1), phi(s1_prime))))
        
        # Optimize f_net by minimizing loss_f
        f_net.zero_grad()
        loss_f.backward()
        f_net_optimizer.step()
        
        f_net = network_weight_matrices(f_net, 1)

    # Second loop for optimizing actor and critic networks
    ppo = PPO(actor_net, critic_net)
    ppo.update(sample_buffer)

class PPO:
    def __init__(self, actor_net, critic_net, actor_lr=3e-4, critic_lr=1e-3, lmbda=0.99, clip_epsilon=0.2, k_epochs=4, epochs=100, device='cuda'):
        self.actor_net = actor_net
        self.critic_net = critic_net
        self.actor_optimizer = optim.Adam(actor_net.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(critic_net.parameters(), lr=critic_lr)
        self.lmbda = lmbda
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.device = device
    

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],dtype=torch.float).view(-1, 1).to(self.device)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -dones)
        td_delta = td_target - self.critic(states)
        advantage = self.compute_advantage(self.gamma, self.lmbda,td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1,
                                                            actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
    
    def compute_advantage(gamma, lmbda, td_delta):
        td_delta = td_delta.detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = gamma * lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.tensor(advantage_list, dtype=torch.float)
