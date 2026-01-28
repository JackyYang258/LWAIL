import d4rl
import gym
import numpy as np
from utils import get_normalized_score
def get_episode_returns(env_name):
    # Create the environment
    env = gym.make(env_name)
    
    # Get the dataset
    dataset = d4rl.qlearning_dataset(env)
    
    # Find episode boundaries using terminals/dones_float
    dones_float = np.zeros_like(dataset['rewards'])
    for i in range(len(dones_float) - 1):
        if np.linalg.norm(dataset['observations'][i + 1] - dataset['next_observations'][i]) > 1e-6 or dataset['terminals'][i] == 1.0:
            dones_float[i] = 1
    dones_float[-1] = 1
    
    # Split into episodes
    episode_returns = []
    current_return = 0
    
    for i in range(len(dataset['rewards'])):
        current_return += dataset['rewards'][i]
        if dones_float[i] == 1:
            episode_returns.append(current_return)
            current_return = 0
    
    return episode_returns, env

def calculate_average_normalized_score(env_name):
    episode_returns, env = get_episode_returns(env_name)
    
    # Calculate normalized scores for each episode
    normalized_scores = [env.get_normalized_score(ret) * 100 for ret in episode_returns]
    
    # Calculate average
    avg_normalized_score = np.mean(normalized_scores)
    
    return avg_normalized_score, len(episode_returns)

# Calculate for both datasets
human_score, human_episodes = calculate_average_normalized_score('pen-human-v0')
cloned_score, cloned_episodes = calculate_average_normalized_score('pen-cloned-v0')

print(f"Results:")
print(f"pen-human-v0: Average normalized score = {human_score:.2f} (over {human_episodes} episodes)")
print(f"pen-cloned-v0: Average normalized score = {cloned_score:.2f} (over {cloned_episodes} episodes)")