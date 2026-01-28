import numpy as np
import torch
import gym
import shimmy
import argparse
import os
import d4rl
import torch.nn as nn
import utils
import TD3
import OurDDPG
import DDPG
import warnings
warnings.filterwarnings("ignore")

# import wandb

class FixedStartWrapper(gym.Wrapper):
    def __init__(self, env, fixed_start):
        super().__init__(env)
        self.fixed_start = np.array(fixed_start, dtype=np.float32)
        assert len(self.fixed_start) == 4, "fixed_start must be (x, y, vx, vy)"

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        # 强制覆盖 reset 后的 obs
        obs[:4] = self.fixed_start.copy()
        # 同时修改环境内部状态
        if hasattr(self.env, "sim"):
            # Maze2D 用 mujoco-like sim
            self.env.sim.data.qpos[0] = self.fixed_start[0]
            self.env.sim.data.qpos[1] = self.fixed_start[1]
            self.env.sim.data.qvel[0] = self.fixed_start[2]
            self.env.sim.data.qvel[1] = self.fixed_start[3]
        return obs
    
# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env = FixedStartWrapper(eval_env, fixed_start=[1.0, 1.0, 0.0, 0.0])

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward

class FullyConnectedNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims, activation=nn.ReLU, activate_final=False):
        super(FullyConnectedNet, self).__init__()
        layers = []
        current_dim = input_dim  # Since we will concatenate two inputs
        for dim in hidden_dims:
            layers.append(nn.Linear(current_dim, dim))
            layers.append(activation())
            current_dim = dim
        layers.append(nn.Linear(current_dim, 1))  # Output dimension is 1
        if activate_final:
            layers.append(activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x1, x2 = None):
        # Concatenate the inputs along the feature dimension
        if x2 is None:
            x = x1
        else:
            x = torch.cat((x1, x2), dim=-1)
        return self.net(x)


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--env", default="maze2d-large-dense-v1")          # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=5e5, type=int)   # Max time steps to run environment
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

	file_name = f"{args.policy}_{args.env}_{args.seed}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	if "dm" in args.env:
		# dm_control2gym wraps a dm_control environment to make it compatible with OpenAI gym
		env = gym.make("dm_control/reacher-easy-v0")
	else:
		print(args.env)
		env = gym.make(args.env)
		env = FixedStartWrapper(env, fixed_start=[1.0, 1.0, 0.0, 0.0])
	# Set seeds
	env.seed(args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
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

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"./models/{policy_file}")

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
 
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	evaluations = [eval_policy(policy, args.env, args.seed)]
	# wandb.init(project='intentDICE', entity="team_siqi", config=args, name="td3", mode='online')
	state, done = env.reset(), False
	print("state:", state)
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0

	for t in range(int(args.max_timesteps)):
		
		episode_timesteps += 1

		# Select action randomly or according to policy
		if t < args.start_timesteps:
			action = env.action_space.sample()
		else:
			action = (
				policy.select_action(np.array(state))
				+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
			).clip(-max_action, max_action)

		# Perform action
		next_state, reward, done, _ = env.step(action)
		max_ep_steps = (
			env._max_episode_steps
			if hasattr(env, "_max_episode_steps")
			else env.spec.max_episode_steps
		)

		done_bool = float(done) if episode_timesteps < max_ep_steps else 0


		# Store data in replay buffer
		replay_buffer.add(state, action, next_state, reward, done_bool)

		state = next_state
		episode_reward += reward

		# Train agent after collecting sufficient data
		if t >= args.start_timesteps:
			policy.train(replay_buffer, args.batch_size)

		if done: 
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f} Episode Length: {episode_timesteps}")
			# Reset environment
			state, done = env.reset(), False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1 

		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			evaluations.append(eval_policy(policy, args.env, args.seed))
			# wandb.log({"eval_reward": evaluations[-1]}, step=t)
			np.save(f"./results/{file_name}", evaluations)
			policy.save(f"./models/{file_name}")
	eval_policy(policy, args.env, args.seed)
	policy.load(f"./models/{file_name}")
	eval_policy(policy, args.env, args.seed)
