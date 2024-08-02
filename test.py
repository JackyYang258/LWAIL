from utils import time
time()

import gym

import sys
sys.path.append('/scratch/bdaw/kaiyan289/intentDICE')
from d4rl_uitls import make_env, get_dataset



env = make_env('maze2d-open-dense-v0')  # Change to your desired D4RL environment
expert_dataset = get_dataset(env)

print(expert_dataset)
terminals = expert_dataset['terminals']
dones_float = expert_dataset['dones_float']

import numpy as np
terminals = np.array(terminals)
dones_float = np.array(dones_float)

# Get indices where terminals or dones_float are 1
terminal_indices = np.where(terminals == 1)[0]
dones_float_indices = np.where(dones_float == 1)[0]

# Get unique indices (in case some might overlap)
all_indices = np.unique(np.concatenate((terminal_indices, dones_float_indices)))

# Filter the dataset
filtered_dataset = {
    'observations': np.array(expert_dataset['observations'])[all_indices],
    'actions': np.array(expert_dataset['actions'])[all_indices],
    'next_observations': np.array(expert_dataset['next_observations'])[all_indices],
    'terminals': terminals[all_indices],
    'dones_float': dones_float[all_indices]
}

# Optional: convert filtered arrays back to lists if necessar

print(filtered_dataset)