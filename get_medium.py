import numpy as np
import os
import gym
from d4rl_utils import get_dataset
import pickle

env_name = "ant-medium-v2"
env = gym.make(env_name)
print("statedim", env.observation_space.shape[0])
dataset = get_dataset(env, env_name)

episode_traj = {key: dataset[key][:990] for key in dataset.keys()}

save_dir = "medium_expert_trajectory"
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, f"{env_name}.pkl")
with open(save_path, 'wb') as f:
    pickle.dump(episode_traj, f)