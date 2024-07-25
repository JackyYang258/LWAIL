from pathlib import Path

import gym
import d4rl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import trange

from .utils import set_seed_everywhere
from .network import PolicyNetwork, FullyConnectedNet, network_weight_matrices, PhiNet
from .d4rl import make_env, get_dataset

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
    f_optimizer.zero_grad()
    loss.backward()
    f_optimizer.step()

def main(args):
    # initialize environment
    env = gym.make(args.env_name)  # Change to your desired D4RL environment
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    env = make_env(args.env_name)
    expert_dataset = get_dataset(env)
    
    set_seed_everywhere(args.seed)
    
    # set up networks
    hidden_dims = list(map(int, args.hidden_dim.split(',')))
    policy_net = PolicyNetwork(state_dim, action_dim)
    f_net = FullyConnectedNet(state_dim * 2, hidden_dims)
    f_net = network_weight_matrices(f_net, 1)
    
    phi_net = PhiNet(icvf_hidden_dims)
    phi_net.load_state_dict(torch.load(args.icvf_path))
    for param in phi_net.parameters():
        param.requires_grad = False
    
    if args.optimizer == 'sgd':
        policy_optimizer = torch.optim.SGD(policy_net.parameters(), lr=args.lr, momentum=0.9)
        f_optimizer = torch.optim.SGD(f_net.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer == 'adam':
        policy_optimizer = torch.optim.Adam(policy_net.parameters(), lr=args.lr)
        f_optimizer = torch.optim.Adam(f_net.parameters(), lr=args.lr)
    else:
        raise NotImplementedError()
    
    # train
    for step in trange(args.n_episode):
        state = env.reset()
        trajectory = []
        for t in range(args.batch_size):
            action, _ = policy_net(state)
            next_state, _, done, _ = env.step(action)
            trajectory.append((state, action, next_state))
            state = next_state
            if done:
                break
        train(policy_net, f_net, policy_optimizer, f_optimizer, trajectory, expert_dataset) # to be modified
        # todo:evaluation
        f_net = network_weight_matrices(f_net, 1)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--env_name', default='hopper-medium-v2')
    parser.add_argument('--lr', default=1e-3)
    parser.add_argument('--n_steps', default=10**6)
    parser.add_argument('--batch_size', default=256)
    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--seed', default=0)
    parser.add_argument('--log_dir', default='logs')
    parser.add_argument('--icvf_path', default='logs')
    
    parser.add_argument('--hidden_dim', default="256,256")
    main(parser.parse_args())
