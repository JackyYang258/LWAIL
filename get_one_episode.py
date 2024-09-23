import numpy as np
import os
import gym
from d4rl_utils import get_dataset
import pickle  # To store trajectories

def save_first_episode(env_name, save_dir):
    # Create environment
    env = gym.make(env_name)

    # Load dataset
    dataset = get_dataset(env, env_name)

    # Get the indices where episode ends (dones_float == 1)
    done_indices = np.where(dataset['dones_float'] == 1)[0]

    if len(done_indices) == 0:
        print(f"No complete episodes in {env_name}. Skipping...")
        return
    
    # Get the trajectory for the first episode
    first_episode_end = done_indices[0] + 1  # +1 to include the final state
    first_episode_trajectory = {key: dataset[key][:first_episode_end] for key in dataset.keys()}

    # Calculate the sum of rewards for the first episode
    reward_sum = np.sum(first_episode_trajectory['rewards'])
    print(f"Sum of rewards for the first episode of {env_name}: {reward_sum}")
    
    # Save the trajectory to a file
    save_path = os.path.join(save_dir, f"{env_name}_first_episode.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(first_episode_trajectory, f)
    
    print(f"Saved first episode of {env_name} to {save_path}")

# List of environment names (Example)
env_names = ["hopper-random-v2"]

# Directory to save trajectories
save_dir = "one_expert_trajectory"
os.makedirs(save_dir, exist_ok=True)

# Loop through environments and save the first episode
for env_name in env_names:
    save_first_episode(env_name, save_dir)
