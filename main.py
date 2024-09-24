from utils import time
time()

import gym
import d4rl
import sys
from utils import set_seed_everywhere, print_args
from d4rl_utils import make_env, get_dataset
from core import Agent
import wandb
import os
import pickle

def main(args):
    time()
    # initialize environment
    env = gym.make(args.env_name)  # Change to your desired D4RL environment
    set_seed_everywhere(env, args.seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    if args.one_episode:
        savedir = "one_expert_trajectory"
        file_path = os.path.join(savedir, args.env_name + ".pkl")
        with open(file_path, 'rb') as f:
            expert_dataset = pickle.load(f)
    else:
        expert_dataset = get_dataset(env, args.env_name)
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
    
    time()
    
    wandb_name = args.env_name.split('-')[0] + "_seed" + str(args.seed)
    wandb.init(project='intentDICE', entity="team_siqi", config=args, name=args.wandb_name, mode='online')
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

    parser.add_argument('--icvf_path', type=str, default="/home/kaiyan3/siqi/IntentDICE/modelhopper.pt", help='Path to the ICVF model checkpoint.')
    parser.add_argument('--using_icvf', action='store_true', help='Flag to indicate whether to use ICVF.')
    parser.add_argument('--state_action', action='store_true', help='Flag to indicate whether to use only state.')
    parser.add_argument('--update_everystep', action='store_true', help='Flag to update at every step.')
    parser.add_argument('--one_episode', action='store_true', help='Flag to update at every step.')
    parser.add_argument('--minus', action='store_true', help='Flag to update at every step.')
    
    # Important Training arguments
    parser.add_argument('--max_training_timesteps', type=int, default=2000000, help='Maximum number of timesteps for training.')
    parser.add_argument('--start_timesteps', type=int, default=1e4, help='Number of timesteps to start training the agent.')
    parser.add_argument('--f_epoch', type=int, default=50, help='Number of epochs for training the function network.')
    parser.add_argument('--agent_epoch', type=int, default=80, help='Number of epochs for PPO training.')
    parser.add_argument('--reward_coeff', type=float, default=1.0, help='Coefficient for the reward term.')
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
