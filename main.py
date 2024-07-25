from pathlib import Path

import gym
import d4rl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import trange

import sys
sys.path.append('/scratch/bdaw/kaiyan289/intentDICE')
from utils import set_seed_everywhere, select_stochastic_action
from network import PolicyNetwork, FullyConnectedNet, network_weight_matrices, PhiNet
from d4rl_uitls import make_env, get_dataset
from buffer import ReplayBuffer

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

def collect_trajectory(env, policy_net, buffer, max_steps):
    state = env.reset()
    for _ in range(max_steps):
        action, probs = select_stochastic_action(policy_net, state)
        next_state, reward, done, _ = env.step(action)
        transition = (state, next_state)
        buffer.add(transition)
        state = next_state

        if done:
            break

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
    
    buffer = ReplayBuffer(max_size=100000)
    
    # train
    for step in trange(args.n_episode):
        collect_trajectory(env, policy_net, buffer, args.n_steps)
        train(policy_net, f_net, policy_optimizer, f_optimizer, buffer, expert_dataset, args.batch_size) # to be modified
        # todo:evaluation
        f_net = network_weight_matrices(f_net, 1)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--env_name', default='hopper-medium-v2')
    parser.add_argument('--lr', default=1e-3)
    parser.add_argument('--n_episodes', default=10**6)
    parser.add_argument('--batch_size', default=256)
    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--seed', default=0)
    parser.add_argument('--log_dir', default='logs')
    parser.add_argument('--icvf_path', default='logs')
    parser.add_argument('--n_steps', default=1000)
    
    parser.add_argument('--hidden_dim', default="256,256")
    main(parser.parse_args())
