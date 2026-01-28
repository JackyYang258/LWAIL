from torchvision.models import resnet18
import torch.nn as nn
import gym
import metaworld
import random
import os
import torch
from tqdm import tqdm
from collections import deque
import numpy as np
from gym.wrappers import TimeLimit
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
import torchvision
os.environ['MUJOCO_GL'] = 'osmesa'

SEED = 42
random.seed(SEED)

def get_frozen_resnet18():
    net = resnet18(pretrained=True)
    
    for param in net.parameters():
        net.requires_grad = False
    
    backbone = nn.Sequential(*list(net.children())[:-1], nn.Flatten())
    
    return backbone.to('cuda:0')
    
# net = get_frozen_resnet18()

# print(net)

# LICENSE file in the root directory of this source tree.

class MetaWorldWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super().__init__(env)
        self.env = env
        self.cfg = cfg
        self._num_frames = cfg.get("frame_stack", 1)
        self._frames = deque([], maxlen=self._num_frames)
        self.resnet = get_frozen_resnet18()
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(512, )
        )
        self.action_space = self.env.action_space
        self.camera_name = "corner2"
        self.env.model.cam_pos[2] = [0.75, 0.075, 0.7]
        self.resize = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)), torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        

    @property
    def state(self):
        state = self._state_obs.astype(np.float32)
        return np.concatenate((state[:4], state[18 : 18 + 4]))

    def _get_pixel_obs(self):
        return self.render(
        #width=self.cfg['img_size'], height=self.cfg['img_size']
        ).transpose(
            2, 0, 1
        )

    def _stacked_obs(self):
        assert len(self._frames) == self._num_frames
        res = torch.cat(list(self._frames), dim=0)
        if len(res.shape) == 3: res = res.unsqueeze(0)
        return res 


    def reset(self):
        self.env.reset()
        obs = self.env.step(np.zeros_like(self.env.action_space.sample()))[0].astype(
            np.float32
        )
        self._state_obs = obs
        obs = self._get_pixel_obs()
        for _ in range(self._num_frames):
            self._frames.append(self.resize(torch.from_numpy(obs).to('cuda:0').float()))
        with torch.no_grad():
            return self.resnet(self._stacked_obs()).squeeze().detach().cpu().numpy()

    def step(self, action):
        reward = 0
        for _ in range(self.cfg['action_repeat']):
            obs, r, term, trun, info = self.env.step(action)
            reward += r
        obs = obs.astype(np.float32)
        self._state_obs = obs
        obs = self._get_pixel_obs()
        self._frames.append(self.resize(torch.from_numpy(obs).to('cuda:0').float()))
        # reward = float(info["success"]) - 1.0
        with torch.no_grad():
            return self.resnet(self._stacked_obs()).squeeze().detach().cpu().numpy(), reward, term or trun, info

    def render(self, mode="rgb_array"):
        return self.env.render(
            # resolution=(width, height), 
            # camera_name=self.camera_name
        ).copy()

    def observation_spec(self):
        return self.observation_space

    def action_spec(self):
        return self.action_space

    def __getattr__(self, name):
        return getattr(self._env, name)

import importlib

def get_policy_class(env_name):
    # Convert env_name to the expected module and class name formats
    module_name = env_name.replace("-", "_")
    class_name = "Sawyer" + "".join([part.capitalize() for part in env_name.split("-")]) + "V2Policy"

    # Construct the module path
    module_path = f"metaworld.policies.sawyer_{module_name}_v2_policy"

    # Import the module and get the class
    module = importlib.import_module(module_path)
    policy_class = getattr(module, class_name)

    return policy_class


def make_metaworld_env(env_name, cfg):
    env_id = env_name + "-v2-goal-observable"
    
    env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_id](seed=SEED,render_mode='rgb_array')
    env._freeze_rand_vec = False
    env = MetaWorldWrapper(env, cfg)
    
    env = TimeLimit(env, max_episode_steps=cfg['max_steps'])
    
    # cfg.state_dim = 8
    return env

cfg = {'max_steps': 500, 'img_size': 224, 'action_repeat': 1}
env_name = "pick-place"


def get_metaenv_traj(env_name, N=100, expert_flag=True):
    
    env = make_metaworld_env(env_name, cfg)
    policy = get_policy_class(env_name)()
    
    dataset = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'groundtruth_state': [],
        'groundtruth_nextstate': [],
        'next_observations': [],
        'dones': [],
    }
    
    time_step = 0
    done_positions = []

    for _ in tqdm(range(N)):
        state = env.reset()
        total_reward = 0 
        for step in range(1, env._max_episode_steps + 1):
            if expert_flag:
                action = policy.get_action(env.env._state_obs)
            else:
                action = env.action_space.sample()
            
            dataset['groundtruth_state'].append(env.env._state_obs)
            
            next_state, reward, done, _ = env.step(action)
            
            dataset['groundtruth_nextstate'].append(env.env._state_obs)
            total_reward += reward
            # Append the data to the dictionary
            dataset['observations'].append(state)
            # print(state.shape)
            dataset['actions'].append(action)
            dataset['rewards'].append(reward)
            dataset['next_observations'].append(next_state)
            dataset['dones'].append(done)
            
            state = next_state
            time_step += 1

            if done:
                # Print the time step where 'done' is True
                print(f"'done' at time step: {time_step}, 'total reward': {total_reward}")
                done_positions.append(time_step)
                break


    # Convert lists to numpy arrays for consistency, like in d4rl
    dataset['observations'] = np.array(dataset['observations'])
    dataset['actions'] = np.array(dataset['actions'])
    dataset['rewards'] = np.array(dataset['rewards'])
    dataset['next_observations'] = np.array(dataset['next_observations'])
    dataset['dones'] = np.array(dataset['dones'])

    env.close()
    return dataset, done_positions
    """
    success_count = 0
    
    for episode in range(N):
        obs = env.reset()
        done = False
        episode_data = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'next_observations': [],
            'dones': [],
            'infos': []
        }
        
        while not done:
            action = policy.get_action(obs)
            next_obs, reward, done, info = env.step(action)
            
            # Store transition
            episode_data['observations'].append(obs)
            episode_data['actions'].append(action)
            episode_data['rewards'].append(reward)
            episode_data['next_observations'].append(next_obs)
            episode_data['dones'].append(done)
            episode_data['infos'].append(info)
            
            obs = next_obs
            
            if done:
                if info.get('success', False):
                    success_count += 1
                break
        
        # Add episode data to dataset
        for key in dataset:
            if key in episode_data:
                dataset[key].extend(episode_data[key])
        
        # Add terminal flags (True at the end of each episode)
        terminals = [False] * (len(episode_data['observations']) - 1) + [True]
        dataset['terminals'].extend(terminals)
        
        print(f"Episode {episode + 1}/{N} completed. Success: {info.get('success', False)}")
    
    # Convert lists to numpy arrays
    for key in dataset:
        dataset[key] = np.array(dataset[key])
    
    success_rate = success_count / N
    print(f"Expert trajectory collection completed. Success rate: {success_rate:.2f}")
    
    return dataset
    """
    

"""
def get_metaenv_random_traj(env_name, N=100):
    pass
"""
#env = make_metaworld_env(env_name, cfg)
#print(env.reset().shape)

if __name__ == "__main__":
    import pickle
    env_name = "pick-place"
    
    expert_traj = get_metaenv_traj("pick-place", N=50, expert_flag=True)
    file_name = "/home/kaiyan3/siqi/IntentDICE/metaworld_" + env_name + ".expert50.pkl"
    with open(file_name, 'wb') as f:
        pickle.dump(expert_traj, f)
    # random_traj = get_metaenv_traj("pick-place", N=1000, expert_flag=False)
    # file_name = "/home/kaiyan3/siqi/IntentDICE/metaworld_" + env_name + ".random.pkl"
    # with open(file_name, 'wb') as f:
    #     pickle.dump(random_traj)
    print(len(expert_traj['observations']), len(episode_data['actions']))
    
    