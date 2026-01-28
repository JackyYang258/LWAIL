import numpy as np
import torch
import gymnasium as gym
import shimmy
import argparse
import os
import d4rl
import torch.nn as nn
import utils
import TD3
import OurDDPG
import DDPG
import pickle


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env = gym.wrappers.FlattenObservation(eval_env)
	avg_reward = 0.
	for _ in range(eval_episodes):
		(state, info), done = eval_env.reset(), False
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, terminated, truncated, _ = eval_env.step(action)
			done = terminated or truncated
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward

def generate_expert_data(policy, env_name, seed, max_data_size=500000):
    eval_env = gym.make(env_name)
    eval_env = gym.wrappers.FlattenObservation(eval_env)

    observations, actions, next_observations, rewards, terminals = [], [], [], [], []
    total_data_count = 0
    total_reward = 0 

    while total_data_count < max_data_size:
        (state, info), done = env.reset(), False
        episode_reward = 0  
        while not done and total_data_count < max_data_size:
            action = policy.select_action(np.array(state))
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            observations.append(state)
            actions.append(action)
            rewards.append(reward)
            next_observations.append(next_state)
            terminals.append(float(done))  # 转换为 float 0 或 1

            state = next_state
            episode_reward += reward  # 增加本轮奖励
            total_data_count += 1

            if total_data_count >= max_data_size:
                break

        total_reward += episode_reward  # 增加总奖励

    avg_reward = total_reward / (total_data_count // 1000)
    print(f"Collected {total_data_count} samples.")
    print(f"Average Reward: {avg_reward:.3f}")
    
    # 转换为 numpy 数组
    dataset = {
        'observations': np.array(observations),
        'actions': np.array(actions),
        'next_observations': np.array(next_observations),
        'rewards': np.array(rewards),
        'terminals': np.array(terminals),
    }
    
    return dataset, avg_reward


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--env", default="dm_control/humanoid-stand-v0")          # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e3, type=int)   # Max time steps to run environment
	parser.add_argument("--expl_noise", default=0.1, type=float)    # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99, type=float)     # Discount factor
	parser.add_argument("--tau", default=0.005, type=float)         # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	args = parser.parse_args()

	env_name = args.env.split('/')[1]
	file_name = f"{args.policy}_{env_name}_{args.seed}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	env = gym.make(args.env)
	env = gym.wrappers.FlattenObservation(env)
	torch.manual_seed(args.seed)
	
	print(env.observation_space)
	print(env.action_space)
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
	}

	# Initialize policy
	if args.policy == "TD3":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		policy = TD3.TD3(**kwargs)
	elif args.policy == "OurDDPG":
		policy = OurDDPG.DDPG(**kwargs)
	elif args.policy == "DDPG":
		policy = DDPG.DDPG(**kwargs)

	policy.load(f"./models/{file_name}")
	eval_policy(policy, args.env, args.seed)
	dataset, avg_reward = generate_expert_data(policy, args.env, args.seed, max_data_size=500000)
 	#save dataset
	with open(f'dataset/{env_name}.pkl', 'wb') as f:
		pickle.dump(dataset, f)
 
	import matplotlib.pyplot as plt
	x = dataset['observations'][:1000][:,0]
	y = dataset['observations'][:1000][:,1]
	plt.scatter(x, y, c='blue', marker='o')

	plt.title("Expert States Plot (from PyTorch Tensor)")
	plt.xlabel("X Coordinate")
	plt.ylabel("Y Coordinate")
	plt.savefig(f'visual/points')  # Add time_step to the filename
	plt.close()
