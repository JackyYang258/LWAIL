import d4rl
import gym
import numpy as np

import time
import os

def make_env(env_name: str):
    env = gym.make(env_name)
    env = EpisodeMonitor(env)
    return env

def get_dataset(env,
                env_name: str,
                clip_to_eps: bool = True,
                eps: float = 1e-5,):
    # save_dir = "/home/kaiyan3/siqi/IntentDICE/d4rl_datasets"
    # save_path = os.path.join(save_dir, f"{env_name}.npz")
    # if 'maze' in env_name:
    #     save_path = "/home/kaiyan3/siqi/IntentDICE/d4rl_datasets/maze2d_expert_dataset.npz"
    # print(f"Loading dataset for path: {save_path}")
    # if not os.path.exists(save_path):
    #     raise FileNotFoundError(f"Dataset file not found for task: {save_path}")
    # data = np.load(save_path, allow_pickle=True)
    # dataset = {
    #     'observations': data['observations'],
    #     'actions': data['actions'],
    #     'rewards': data['rewards'],
    #     'terminals': data['terminals'],
    #     'next_observations': data['next_observations'] if 'next_observations' in data else None
    # }
    if 'maze2d-open' in env_name:
        print("Loading maze dataset")
        dataset = np.load("/home/kaiyan3/siqi/IntentDICE/d4rl_datasets/maze2d_expert_dataset.npz", allow_pickle=True)
        dataset = {
            'observations': dataset['observations'],
            'actions': dataset['actions'],
            'rewards': dataset['rewards'],
            'terminals': dataset['terminals'],
            'next_observations': dataset['next_observations'] 
        }
    else:
        dataset = d4rl.qlearning_dataset(env)
    print(dataset['observations'].shape)
    print(dataset['next_observations'].shape)

    import matplotlib.pyplot as plt
    x = dataset['observations'][:1000][:,0]
    y = dataset['observations'][:1000][:,1]
    plt.scatter(x, y, c='blue', marker='o')

    plt.title("Expert States Plot (from PyTorch Tensor)")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.savefig(f'visual/data_points')  # Add time_step to the filename
    plt.close()
 
    if clip_to_eps:
        lim = 1 - eps
        dataset['actions'] = np.clip(dataset['actions'], -lim, lim)

    dones_float = np.zeros_like(dataset['rewards'])

    for i in range(len(dones_float) - 1):
        if np.linalg.norm(dataset['observations'][i + 1] -
                            dataset['next_observations'][i]
                            ) > 1e-6 or dataset['terminals'][i] == 1.0:
            dones_float[i] = 1
        else:
            dones_float[i] = 0

    dones_float[-1] = 1

    dataset['dones_float'] = dones_float

    # Print the locations where dones_float is equal to 1
    # done_indices = np.where(dones_float == 1)[0]
    # print(f"Locations where dones_float is 1: {done_indices}")
    
    return dataset

class EpisodeMonitor(gym.ActionWrapper):
    """A class that computes episode returns and lengths."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action: np.ndarray):
        observation, reward, done, info = self.env.step(action)

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info["total"] = {"timesteps": self.total_timesteps}

        if done:
            info["episode"] = {}
            info["episode"]["return"] = self.reward_sum
            info["episode"]["length"] = self.episode_length
            info["episode"]["duration"] = time.time() - self.start_time

            if hasattr(self, "get_normalized_score"):
                info["episode"]["normalized_return"] = (
                    self.get_normalized_score(info["episode"]["return"]) * 100.0
                )

        return observation, reward, done, info

    def reset(self) -> np.ndarray:
        self._reset_stats()
        return self.env.reset()