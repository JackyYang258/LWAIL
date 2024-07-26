import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

def train(expert_buffer, sample_buffer, f_net, actor_net, critic_net, m, phi, f_net_optimizer, actor_net_optimizer, critic_net_optimizer, on_policy_rl_algorithm):
    for iter_f in range(m):
        # Draw expert trajectory (s_1, s_1') from expert_buffer
        s1, s1_prime = expert_buffer.sample()
        
        # Draw sample trajectory (s_2, s_2') from sample_buffer
        s2, s2_prime = sample_buffer.sample()
        
        # Calculate the loss
        loss_f = (torch.mean(f_net(phi(s2), phi(s2_prime))) - 
                  torch.mean(f_net(phi(s1), phi(s1_prime))))
        
        # Optimize f_net by minimizing loss_f
        f_net.zero_grad()
        loss_f.backward()
        f_net_optimizer.step()

    # Second loop for optimizing actor and critic networks
    for iter_a in range(m):
        # Draw sample trajectory (s_2, a_2, s_2') from sample_buffer
        s, a, s_prime = sample_buffer.sample_with_action()
        
        # Calculate reward as f(s_2, s_2')
        reward = f_net(s, s_prime).detach()
        
        # Optimize actor and critic using on-policy RL algorithm
        on_policy_rl_algorithm(actor_net, critic_net, s, a, s_prime, reward)

class PPO:
    def __init__(self, actor_net, critic_net, actor_lr=3e-4, critic_lr=1e-3, gamma=0.99, clip_epsilon=0.2, k_epochs=4):
        self.actor_net = actor_net
        self.critic_net = critic_net
        self.actor_optimizer = optim.Adam(actor_net.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(critic_net.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.k_epochs = k_epochs

    def compute_returns_and_advantages(self, rewards, values, next_values, dones):
        returns = []
        advantages = []
        advantage = 0
        for t in reversed(range(len(rewards))):
            td_error = rewards[t] + (1 - dones[t]) * self.gamma * next_values[t] - values[t]
            advantage = td_error + (1 - dones[t]) * self.gamma * advantage
            advantages.insert(0, advantage)
            returns.insert(0, advantage + values[t])
        return returns, advantages

    def update(self, states, actions, rewards, next_states, dones):
        values = self.critic_net(states).detach()
        next_values = self.critic_net(next_states).detach()
        returns, advantages = self.compute_returns_and_advantages(rewards, values, next_values, dones)
        returns = torch.tensor(returns).float()
        advantages = torch.tensor(advantages).float()
        old_log_probs = Categorical(self.actor_net(states)).log_prob(actions)

        for _ in range(self.k_epochs):
            log_probs = Categorical(self.actor_net(states)).log_prob(actions)
            ratios = torch.exp(log_probs - old_log_probs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(self.critic_net(states), returns)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

def on_policy_rl_algorithm(actor_net, critic_net, s, a, s_prime, reward):
    # Calculate the loss
    ppo = PPO(actor_net, critic_net)
    ppo.update(s, a, s_prime, reward)
    
    return actor_net, critic_net