from utils import log_time
log_time()

import gym
import d4rl
import sys
from utils import set_seed_everywhere, print_args
from d4rl_utils import make_env, get_dataset
from core import Agent
from network import Contrastive_PD, ContrastiveEncoder
import numpy as np
import wandb
import os
import pickle
from d4rl_utils import FixedStartWrapper


def main(args):
    log_time()
    # initialize environment
    env = gym.make(args.env_name)
    if 'maze2d' in args.env_name:
        env = FixedStartWrapper(env)
    set_seed_everywhere(env, args.seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # load dataset
    if args.expert_episode == "one":
        savedir = "one_expert_trajectory"
        file_path = os.path.join(savedir, args.env_name + args.dataset_suffix + ".pkl")
        with open(file_path, 'rb') as f:
            expert_dataset = pickle.load(f)
    elif args.expert_episode == "five":
        savedir = "5_expert_trajectory"
        file_path = os.path.join(savedir, args.env_name + ".pkl")
        with open(file_path, 'rb') as f:
            expert_dataset = pickle.load(f)
            
    elif args.expert_episode == "antmaze":
        savedir = "antmaze_expert_trajectory"
        file_path = os.path.join(savedir, args.env_name + ".pkl")
        with open(file_path, 'rb') as f:
            expert_dataset = pickle.load(f)
            
    elif args.expert_episode == "uncom5":
        savedir = "uncomplete5_expert_trajectory"
        file_path = os.path.join(savedir, args.env_name + ".pkl")
        with open(file_path, 'rb') as f:
            expert_dataset = pickle.load(f)
    elif args.expert_episode == "uncom10":
        savedir = "uncomplete10_expert_trajectory"
        file_path = os.path.join(savedir, args.env_name + ".pkl")
        with open(file_path, 'rb') as f:
            expert_dataset = pickle.load(f)
    elif args.expert_episode == "uncom20":
        savedir = "uncomplete20_expert_trajectory"
        file_path = os.path.join(savedir, args.env_name + ".pkl")
        with open(file_path, 'rb') as f:
            expert_dataset = pickle.load(f)
            
    elif args.expert_episode == "multiple":
        expert_dataset = get_dataset(env, args.env_name)
    elif args.expert_episode == "sample":
        sampled_data = {}
        expert_dataset = get_dataset(env, args.env_name)
        for key, array in expert_dataset.items():
            sample_size = 10000
            size = array.shape[0]
            random_indices = np.random.choice(size, sample_size, replace=False)
            if array.ndim == 1: 
                sampled_data[key] = array[random_indices]
            elif array.ndim == 2:  
                sampled_data[key] = array[random_indices, :]
            else:
                raise ValueError("Unsupported array dimension. Only 1D or 2D arrays are supported.")
        expert_dataset = sampled_data
    elif args.expert_episode == "mismatchant":
        demo_file = f"mismatch/ant.pkl"
        demo = pickle.load(open(demo_file, 'rb'))
        expert_obs = np.array(demo['observations'][:1000])
        expert_actions = np.array(demo['actions'][:1000])
        expert_next_obs = np.array(demo['next_observations'][:1000])
        expert_dataset = {'observations': expert_obs, 'actions': expert_actions, 'next_observations': expert_next_obs}
    elif args.expert_episode == "mismatchhalfcheetah":
        demo_file = f"mismatch/halfcheetah.pkl"
        demo = pickle.load(open(demo_file, 'rb'))
        expert_obs = np.array(demo['observations'][0])
        expert_actions = np.array(demo['actions'][0])
        expert_next_obs = np.array(demo['next_observations'][0])
        expert_dataset = {'observations': expert_obs, 'actions': expert_actions, 'next_observations': expert_next_obs}
    print("dataset_number", args.expert_episode)
    print("dataset size:", expert_dataset['observations'].shape[0])
    
    #print informations about the environment
    print('state_dim:', state_dim)
    print('action_dim:', action_dim)
    print('observation_space:', env.observation_space)
    print('action_space:', env.action_space)
    
    print_args(args)
    if args.using_icvf:
        print("Using ICVF")
    if args.update_everystep:
        print("Update every step")
    
    log_time()
    
    wandb_name = "LWAIL_"+ args.wandb_name + "_" + args.env_name.split('-')[0] + "_" + str(args.seed)
    wandb.init(project='intentDICE', entity="team_siqi", config=args, name=wandb_name, mode='online')
    agent = Agent(state_dim, action_dim, env, expert_dataset, args)
    print("======== start training ==========")
    agent.train()
    wandb.finish()



if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser() 
    
    # Environment
    parser.add_argument('--env_name', type=str, default='hopper-expert-v2', help='Name of the environment to use.')
    parser.add_argument('--wandb_name', type=str, default='maze', help='Name of the environment to use.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility.')

    parser.add_argument('--icvf_path', type=str, help='Path to the ICVF model checkpoint.')
    parser.add_argument('--cuda', type=str)
    parser.add_argument('--using_icvf', action='store_true', help='Flag to indicate whether to use ICVF.')
    parser.add_argument('--using_pwdice', action='store_true')
    parser.add_argument('--state_action', action='store_true', help='Flag to indicate whether to use only state.')
    parser.add_argument('--update_everystep', action='store_true', help='Flag to update at every step.')
    parser.add_argument('--expert_episode', default="one")
    parser.add_argument('--minus', action='store_true')
    parser.add_argument('--curl', action='store_true')
    parser.add_argument('--dataset_num', type=str, default="")
    parser.add_argument('--dataset_quality', type=str, default="")
    parser.add_argument('--dataset_suffix', type=str, default="")
    
    # Important Training arguments
    parser.add_argument('--max_training_timesteps', type=int, default=2000000, help='Maximum number of timesteps for training.')
    parser.add_argument('--start_timesteps', type=int, default=1e4, help='Number of timesteps to start training the agent.')
    parser.add_argument('--f_epoch', type=int, default=50, help='Number of epochs for training the function network.')
    parser.add_argument('--lr_f', type=float, default=1e-3, help='Learning rate for the function network.')
    parser.add_argument('--lr_actor', type=float, default=3e-4, help='Learning rate for the actor network.')
    parser.add_argument('--lr_critic', type=float, default=1e-3, help='Learning rate for the critic network.')
    parser.add_argument('--alpha', type=int, default=10)
    
    parser.add_argument('--hidden_dim', type=str, default='64,64', help='Comma-separated list of hidden dimensions for the network.')
    parser.add_argument('--downstream', type=str, default='ppo')
    parser.add_argument('--max_ep_len', type=int, default=1000, help='Maximum length of each episode.')
    parser.add_argument('--update_timestep', type=int, default=4000, help='Number of timesteps between updates.')
    parser.add_argument('--eval_freq', type=int, default=10000, help='frequency of evaluation.')
    
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor for rewards.')
    parser.add_argument('--eps_clip', type=float, default=0.2, help='Clip parameter for PPO.')
    main(parser.parse_args())
