import numpy as np
import os
import gym
from d4rl_utils import get_dataset
import pickle  # To store trajectories

def save_top_rewards_episodes(env_name, save_dir, top_k=20, skip=20):
    # Create environment
    env = gym.make(env_name)

    # Load dataset
    dataset = get_dataset(env, env_name)

    # Get the indices where episode ends (dones_float == 1)
    done_indices = np.where(dataset['dones_float'] == 1)[0]

    if len(done_indices) < 50:
        print(f"Less than 20 complete episodes in {env_name}. Skipping...")
        return
    
    # Get the first 20 episodes
    num_episodes = min(50, len(done_indices))
    episodes = []
    rewards = []

    start_idx = 0
    for i in range(num_episodes):
        end_idx = done_indices[i] + 1  # +1 to include the final state
        episode_trajectory = {key: dataset[key][start_idx:end_idx] for key in dataset.keys()}
        episode_reward_sum = np.sum(episode_trajectory['rewards'])
        
        # Store the trajectory and corresponding reward
        episodes.append((episode_trajectory, episode_reward_sum))
        
        # Update the start index for the next episode
        start_idx = end_idx

    # Sort episodes by reward in descending order
    episodes.sort(key=lambda x: x[1], reverse=True)

    # Get the top k episodes with the highest rewards
    top_episodes = episodes[:top_k]

    for i, (episode_trajectory, reward_sum) in enumerate(top_episodes):
        print(f"Sum of rewards for top {i+1} episode of {env_name}: {reward_sum}")
    # Combine the top k episodes into one complete episode
    combined_episode = {}
    for key in dataset.keys():
        # Initialize the combined episode with empty arrays
        combined_episode[key] = np.empty((0,) + dataset[key].shape[1:], dtype=dataset[key].dtype)
        for episode in top_episodes:
            # Append every skip-th element to the combined episode
            combined_episode[key] = np.append(combined_episode[key], episode[0][key][::skip], axis=0)

    # Save the combined episode to a single file
    save_path = os.path.join(save_dir, f"{env_name}.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(combined_episode, f)
    
    # Print the combined episode dictionary
    print("Combined episode dictionary:")
    for key, value in combined_episode.items():
        print(f"{key}: {value.shape}...")  # Print shape and first few elements

    print(f"Saved combined top {top_k} reward episodes of {env_name} to {save_path}")

# List of environment names (Example)
env_names = ["hopper-expert-v2", "halfcheetah-expert-v2", "walker2d-expert-v2", "ant-expert-v2"]

# Directory to save trajectories
save_dir = "uncomplete20_expert_trajectory"
os.makedirs(save_dir, exist_ok=True)

# Loop through environments and save the top k reward episodes
for env_name in env_names:
    save_top_rewards_episodes(env_name, save_dir)