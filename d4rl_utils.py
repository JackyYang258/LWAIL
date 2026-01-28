import d4rl
import gym
import numpy as np

import time
import pickle
import os
import minari
import numpy as np
import gym

class FixedStartWrapper(gym.Wrapper):
    def __init__(self, env, fixed_start=[1.0,1.0,0.0,0.0], noise_scale=0.3):
        """
        Args:
            env: The gym environment.
            fixed_start: The central starting state (x, y, vx, vy).
            noise_scale: Standard deviation of the Gaussian noise to add. 
                         Can be a float (same for all dims) or a list of length 4.
                         Defaults to 0.0 (no noise).
        """
        super().__init__(env)
        self.fixed_start = np.array(fixed_start, dtype=np.float32)
        self.noise_scale = noise_scale
        assert len(self.fixed_start) == 4, "fixed_start must be (x, y, vx, vy)"

    def reset(self, **kwargs):
        # Reset the base environment
        # Note: If using Gym >= v0.26, keep in mind reset returns (obs, info)
        result = super().reset(**kwargs)
        
        # Handle legacy Gym (obs only) vs New Gym (obs, info)
        if isinstance(result, tuple):
            obs = result[0]
        else:
            obs = result

        # Generate random noise: N(0, noise_scale)
        noise = np.random.normal(loc=0.0, scale=self.noise_scale, size=4).astype(np.float32)
        
        # Calculate new noisy start
        noisy_start = self.fixed_start + noise

        # Apply to observation
        obs[:4] = noisy_start.copy()

        # Apply to simulation internals
        if hasattr(self.env, "sim"):
            self.env.sim.data.qpos[0] = noisy_start[0]
            self.env.sim.data.qpos[1] = noisy_start[1]
            self.env.sim.data.qvel[0] = noisy_start[2]
            self.env.sim.data.qvel[1] = noisy_start[3]
            
            # CRITICAL: Propagate changes through the physics engine
            # Without this, internal physics variables (accelerations, sensors) may be stale
            self.env.sim.forward()

        # Return consistent with the API version you are using
        if isinstance(result, tuple):
            return obs, result[1]
        return obs
    
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
    if 'maze2d-' in env_name:
        print("Loading maze dataset")
        file_path = f"/home/kaiyan3/siqi/IntentDICE/multiple_expert_trajectory/{env_name}.pkl"
        with open(file_path, 'rb') as f:
            dataset = pickle.load(f)
        dataset = {
            'observations': dataset['observations'],
            'actions': dataset['actions'],
            'rewards': dataset['rewards'],
            'terminals': dataset['terminals'],
            'next_observations': dataset['next_observations'] 
        }
    elif 'dm_control' in env_name:
        path_name = env_name.split("/")[1]
        file_path = "/home/kaiyan3/siqi/IntentDICE/multiple_expert_trajectory/" + path_name + ".pkl"
        with open(file_path, 'rb') as f:
            dataset = pickle.load(f)
        dataset = {
            'observations': dataset['observations'],
            'actions': dataset['actions'],
            'rewards': dataset['rewards'],
            'terminals': dataset['terminals'],
            'next_observations': dataset['next_observations'] 
        }
    elif 'mujoco' in env_name:
        dataset = minari.load_dataset('mujoco/humanoid/expert-v0')
        obs_list, act_list, next_list, rew_list, done_list = [], [], [], [], []

        for episode in dataset.iterate_episodes():
            obs = episode.observations
            act = episode.actions
            rew = episode.rewards

            done = np.logical_or(
                episode.terminations,
                episode.truncations
            ).astype(bool)

            next_obs = obs[1:]
            done = done[:-1]
            rew = rew[:-1]
            act = act[:-1]
            obs = obs[:-1]

            # 确保所有都是 1D/2D array，不是 scalar
            done = np.asarray(done).reshape(-1)
            rew = np.asarray(rew).reshape(-1, 1)

            obs_list.append(obs)
            next_list.append(next_obs)
            act_list.append(act)
            rew_list.append(rew)
            done_list.append(done)
        dataset = {}
        dataset['observations'] = np.concatenate(obs_list, axis=0)
        dataset['actions'] = np.concatenate(act_list, axis=0)
        dataset['next_observations'] = np.concatenate(next_list, axis=0)
        dataset['rewards'] = np.concatenate(rew_list, axis=0)
        dataset['terminals'] = np.concatenate(done_list, axis=0)
    else:
        dataset = d4rl.qlearning_dataset(env)
    

    # import matplotlib.pyplot as plt
    # x = dataset['observations'][:1000][:,0]
    # y = dataset['observations'][:1000][:,1]
    # plt.scatter(x, y, c='blue', marker='o')

    # plt.title("Expert States Plot (from PyTorch Tensor)")
    # plt.xlabel("X Coordinate")
    # plt.ylabel("Y Coordinate")
    # plt.savefig(f'visual/data_points')  # Add time_step to the filename
    # plt.close()
 
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