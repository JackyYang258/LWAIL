import d4rl
import gym
import numpy as np

import time

def make_env(env_name: str):
    env = gym.make(env_name)
    env = EpisodeMonitor(env)
    return env

def get_dataset(env: gym.Env,
                using_d4rl: bool,
                dataset_path: str,
                clip_to_eps: bool = True,
                eps: float = 1e-5,):
    if using_d4rl:
        dataset = d4rl.qlearning_dataset(env)
    else:
        npzfile = np.load(dataset_path)
        dataset = {key: npzfile[key] for key in npzfile}

    if not 'actions' in dataset:
        return dataset
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