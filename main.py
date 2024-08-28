from utils import time
time()

import gym

import sys
sys.path.append('/scratch/bdaw/kaiyan289/IntentDICE')
from utils import set_seed_everywhere, print_args
from d4rl_uitls import make_env, get_dataset
from core_old import Agent
import wandb

def main(args):
    time()
    # initialize environment
    set_seed_everywhere(args.seed)
    env = gym.make(args.env_name)  # Change to your desired D4RL environment
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    env = make_env(args.env_name)
    expert_dataset = get_dataset(env, using_d4rl=True, dataset_path='/scratch/bdaw/kaiyan289/IntentDICE/dataset/maze2d_expert_dataset.npz')
    print("dataset size:", expert_dataset['observations'].shape[0])
    
    #print informations about the environment
    print('state_dim:', state_dim)
    print('action_dim:', action_dim)
    print('observation_space:', env.observation_space)
    print('action_space:', env.action_space)
    
    print_args(args)
    
    time()
    
    wandb.init(project='intentDICE', config=args, name=args.wandb_name, mode='online')
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

    parser.add_argument('--icvf_path', type=str, default="/scratch/bdaw/kaiyan289/IntentDICE/model/hopper.pt", help='Path to the ICVF model checkpoint.')
    parser.add_argument('--using_icvf', default=True, help='Flag to indicate whether to use ICVF.')
    parser.add_argument('--only_state', default=False)
    parser.add_argument('--update_everystep', default=False)
    
    # Important Training arguments
    parser.add_argument('--max_training_timesteps', type=int, default=1000000, help='Maximum number of timesteps for training.')
    parser.add_argument('--start_timesteps', type=int, default=1e4, help='Number of timesteps to start training the agent.')
    parser.add_argument('--f_epoch', type=int, default=50, help='Number of epochs for training the function network.')
    parser.add_argument('--agent_epoch', type=int, default=50, help='Number of epochs for PPO training.')
    parser.add_argument('--reward_coeff', type=float, default=1.0, help='Coefficient for the reward term.')
    parser.add_argument('--lr_f', type=float, default=1e-3, help='Learning rate for the function network.')
    parser.add_argument('--lr_actor', type=float, default=3e-4, help='Learning rate for the actor network.')
    parser.add_argument('--lr_critic', type=float, default=1e-3, help='Learning rate for the critic network.')
    
    parser.add_argument('--hidden_dim', type=str, default='64,64', help='Comma-separated list of hidden dimensions for the network.')
    parser.add_argument('--max_ep_len', type=int, default=1000, help='Maximum length of each episode.')
    parser.add_argument('--update_timestep', type=int, default=1000, help='Number of timesteps between updates.')
    parser.add_argument('--eval_freq', type=int, default=10000, help='frequency of evaluation.')
    
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor for rewards.')
    parser.add_argument('--eps_clip', type=float, default=0.2, help='Clip parameter for PPO.')
    parser.add_argument('--action_std_decay_frequency', type=int, default=int(2e5), help='Frequency of action standard deviation decay.')
    parser.add_argument('--action_std_decay_rate', type=float, default=0.05, help='Decay rate of the action standard deviation.')
    parser.add_argument('--min_action_std', type=float, default=0.1, help='Minimum value of action standard deviation.')
    parser.add_argument('--action_std_init', type=float, default=0.6, help='Initial standard deviation of action distribution.')
    main(parser.parse_args())
