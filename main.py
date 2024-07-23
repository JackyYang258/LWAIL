import gym
import d4rl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from .utils import set_seed_everywhere
from .network import PolicyNetwork, FullyConnectedNet, network_weight_matrices

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)

class f_Network(nn.Module):
    def __init__(self, state_dim):
        super(f_Network, self).__init__()
        self.fc1 = nn.Linear(state_dim * 2, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        
    def forward(self, state, next_state):
        x = torch.cat((state, next_state), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x).squeeze()

def select_action(policy_net, state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy_net(state)
    action = np.random.choice(len(probs.squeeze()), p=probs.squeeze().detach().numpy())
    return action, probs[:, action].item()

def train(actor_net, f_net, actor_optimizer, f_optimizer, trajectory, expert_trajectory, gamma=0.99):
    states, next_states = zip(*trajectory)
    expert_states, expert_next_states = zip(*expert_trajectory)
    
    states = torch.tensor(states, dtype=torch.float)
    next_states = torch.tensor(next_states, dtype=torch.float)
    expert_states = torch.tensor(expert_states, dtype=torch.float)
    expert_next_states = torch.tensor(expert_next_states, dtype=torch.float)
    
    # Compute target function values
    f_values_sample = f_net(states, next_states)
    f_values_expert = f_net(expert_states, expert_next_states)
    
    # Compute target loss
    loss = f_values_sample.mean() - f_values_expert.mean()
    
    # Update actor using target loss
    actor_optimizer.zero_grad()
    loss.backward()
    actor_optimizer.step()
    
    # Update f network
    target_optimizer.zero_grad()
    loss.backward()
    target_optimizer.step()

def main():
    env = gym.make('hopper-medium-v2')  # Change to your desired D4RL environment
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    set_seed_everywhere(0)
    
    hidden_dims = [128, 128]
    policy_net = PolicyNetwork(state_dim, action_dim)
    f_net = FullyConnectedNet(state_dim * 2, hidden_dims)
    f_net = network_weight_matrices(f_net, 1)
    
    if optim == 'sgd':
        policy_optimizer = torch.optim.SGD(policy_net.parameters(), lr=lr, momentum=0.9)
        f_optimizer = torch.optim.SGD(f_net.parameters(), lr=lr, momentum=0.9)
    elif optim == 'adam':
        policy_optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
        f_optimizer = torch.optim.Adam(f_net.parameters(), lr=lr)
        
    else:
        raise NotImplementedError()
    
    # Example expert trajectory (you should load or generate this)
    expert_trajectory = [
        (np.random.rand(state_dim), np.random.rand(state_dim)) for _ in range(100)
    ]

    for episode in range(1000):
        state = env.reset()
        trajectory = []
        for t in range(100):
            action, _ = select_action(policy_net, state)
            next_state, _, done, _ = env.step(action)
            trajectory.append((state, next_state))
            state = next_state
            if done:
                break
        train(policy_net, f_net, policy_optimizer, f_optimizer, trajectory, expert_trajectory)
        f_net = network_weight_matrices(f_net, 1)

if __name__ == "__main__":
    main()
