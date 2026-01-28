import numpy as np
import os
import gym
from d4rl_utils import get_dataset
import pickle  # To store trajectories

def save_highest_reward_episode(env_name, save_dir):
    # Create environment
    if "dm_control" in env_name or "mujoco" in env_name:
        print("Loading dm_control or mujoco environment")
        print("env_name:", env_name)
    else:
        env = gym.make(env_name)
    
    # Load dataset
    dataset = get_dataset(env, env_name)

    # Get the indices where episode ends (dones_float == 1)
    done_indices = np.where(dataset['dones_float'] == 1)[0]

    if len(done_indices) == 0:
        print(f"No complete episodes in {env_name}. Skipping...")
        return
    
    # Get the first 10 episodes
    num_episodes = min(10, len(done_indices))
    episodes = []
    rewards = []

    start_idx = 0
    for i in range(num_episodes):
        end_idx = done_indices[i] + 1  # +1 to include the final state
        episode_trajectory = {key: dataset[key][start_idx:end_idx] for key in dataset.keys()}
        episode_reward_sum = np.sum(episode_trajectory['rewards'])
        
        # Store the trajectory and corresponding reward
        episodes.append(episode_trajectory)
        rewards.append(episode_reward_sum)
        
        # Update the start index for the next episode
        start_idx = end_idx

    # Print the rewards of the first 10 episodes
    for i, reward_sum in enumerate(rewards):
        print(f"Sum of rewards for episode {i + 1} of {env_name}: {reward_sum}")

    # Find the episode with the highest reward
    max_reward_idx = np.argmax(rewards)
    highest_reward_episode = episodes[max_reward_idx]
    
    trajectory_length = len(highest_reward_episode['rewards'])
    print(f"Length of highest reward trajectory in {env_name}: {trajectory_length} steps")
    # Save the episode with the highest reward to a file
    save_path = os.path.join(save_dir, f"{env_name}.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(highest_reward_episode, f)
    
    print(f"Saved highest reward episode of {env_name} (reward: {rewards[max_reward_idx]}) to {save_path}")

# List of environment names (Example)
env_names = ["maze2d-large-dense-v1"]

# Directory to save trajectories
save_dir = "one_expert_trajectory"
os.makedirs(save_dir, exist_ok=True)

# Loop through environments and save the highest reward episode
for env_name in env_names:
    save_highest_reward_episode(env_name, save_dir)
