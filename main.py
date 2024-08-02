from utils import time
time()

import gym
import torch

import sys
sys.path.append('/scratch/bdaw/kaiyan289/intentDICE')
from utils import set_seed_everywhere
from network import FullyConnectedNet, network_weight_matrices, PhiNet
from d4rl_uitls import make_env, get_dataset
from train import train

def main(args):
    time()
    # initialize environment
    env = gym.make(args.env_name)  # Change to your desired D4RL environment
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    env = make_env(args.env_name)
    expert_dataset = get_dataset(env)
    
    set_seed_everywhere(args.seed)
    #print informations about the environment
    print('state_dim:', state_dim)
    print('action_dim:', action_dim)
    print('observation_space:', env.observation_space)
    print('action_space:', env.action_space)
    
    time()
    # set up networks
    hidden_dims = list(map(int, args.hidden_dim.split(',')))
    f_net = FullyConnectedNet(state_dim * 2, hidden_dims).to('cuda:0')
    f_net = network_weight_matrices(f_net, 1)
    
    if args.using_icvf:
        phi_net = PhiNet(icvf_hidden_dims)
        phi_net.load_state_dict(torch.load(args.icvf_path))
        for param in phi_net.parameters():
            param.requires_grad = False
        print('Using ICVF')
    else:
        phi_net = None
        print('Not using ICVF')
    
    train(expert_dataset, f_net, phi_net, env, args.seed, args.max_ep_len, args.max_training_timesteps,args.update_timestep, args.f_epoch, args.lr_f, args.action_std_decay_frequency, args.action_std_decay_rate, args.min_action_std, state_dim, action_dim, args.lr_actor, args.lr_critic, args.gamma, args.ppo_epochs, args.eps_clip, args.action_std_init)



if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    
    # Environment
    parser.add_argument('--env_name', type=str, default='maze2d-open-dense-v0', help='Name of the environment to use.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility.')

    parser.add_argument('--icvf_path', type=str, default=None, help='Path to the ICVF model checkpoint.')
    parser.add_argument('--using_icvf', default=False, help='Flag to indicate whether to use ICVF.')
    
    # Important Training arguments
    parser.add_argument('--max_training_timesteps', type=int, default=1000000, help='Maximum number of timesteps for training.')
    parser.add_argument('--f_epoch', type=int, default=10, help='Number of epochs for training the function network.')
    parser.add_argument('--ppo_epochs', type=int, default=80, help='Number of epochs for PPO training.')
    parser.add_argument('--lr_f', type=float, default=1e-3, help='Learning rate for the function network.')
    parser.add_argument('--lr_actor', type=float, default=3e-4, help='Learning rate for the actor network.')
    parser.add_argument('--lr_critic', type=float, default=1e-3, help='Learning rate for the critic network.')
    
    
    parser.add_argument('--hidden_dim', type=str, default='256,256', help='Comma-separated list of hidden dimensions for the network.')
    parser.add_argument('--max_ep_len', type=int, default=1000, help='Maximum length of each episode.')
    parser.add_argument('--update_timestep', type=int, default=4000, help='Number of timesteps between updates.')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor for rewards.')
    parser.add_argument('--eps_clip', type=float, default=0.2, help='Clip parameter for PPO.')
    parser.add_argument('--action_std_decay_frequency', type=int, default=int(2e5), help='Frequency of action standard deviation decay.')
    parser.add_argument('--action_std_decay_rate', type=float, default=0.05, help='Decay rate of the action standard deviation.')
    parser.add_argument('--min_action_std', type=float, default=0.1, help='Minimum value of action standard deviation.')
    parser.add_argument('--action_std_init', type=float, default=0.6, help='Initial standard deviation of action distribution.')
    main(parser.parse_args())
