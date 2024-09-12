import d4rl
import gym
import os
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Assuming you have already defined your device (cpu or cuda)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the PhiNet class (as you have it)
class PhiNet(nn.Module):
    def __init__(self, hidden_dims, activation=nn.GELU, activate_final=False):
        super(PhiNet, self).__init__()
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            if i + 1 < len(hidden_dims) or activate_final:
                layers.append(activation())
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x) 

# Initialize the PhiNet and load the state dictionary
icvf_hidden_dims = [11] + [256, 256]
phi_net = PhiNet(icvf_hidden_dims)
print("phi_net:", phi_net)

env_firstname = "hopper"
icvf_path = "/scratch/bdaw/kaiyan289/IntentDICE/model/" + env_firstname + ".pt"
phi_net.load_state_dict(torch.load(icvf_path))
for param in phi_net.parameters():
    param.requires_grad = False
phi_net.to(device)
print('Using ICVF')

os.makedirs('visual', exist_ok=True)

# Create the hopper_expert_v2 environment and load the dataset
env = gym.make('hopper-expert-v2')
dataset = env.get_dataset()

# Extract a trajectory
observations = dataset['observations']
actions = dataset['actions']
next_observations = dataset['next_observations']
rewards = dataset['rewards']
dones = dataset['terminals']

# Find the end of the trajectory
print(dones.argmax())
trajectory_end = 2000
print('Trajectory end:', trajectory_end)

# Select the trajectory
trajectory_observations = observations[:trajectory_end]
trajectory_actions = actions[:trajectory_end]
trajectory_next_observations = next_observations[:trajectory_end]
trajectory_rewards = rewards[:trajectory_end]

# Convert the trajectory observations to torch tensors
trajectory_observations_tensor = torch.tensor(trajectory_observations, dtype=torch.float32).to(device)

# Use phi_net to process the observations
with torch.no_grad():
    processed_observations = phi_net(trajectory_observations_tensor).cpu().numpy()
    
import matplotlib.colors as mcolors
norm = mcolors.Normalize(vmin=trajectory_rewards.min(), vmax=trajectory_rewards.max())

# ================== Visualization Before phi_net (Original Data) ==================

# PCA Visualization (Original)
pca = PCA(n_components=2)
reduced_observations = pca.fit_transform(trajectory_observations)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(reduced_observations[:, 0], reduced_observations[:, 1], c=norm(trajectory_rewards), cmap='viridis', marker='o')
plt.title('2D visualization of the original trajectory (PCA) with normalized rewards')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter, label='Normalized Reward')
plt.savefig('visual/original_trajectory_pca_visualization_with_normalized_rewards.png')
plt.show()

# t-SNE Visualization (Original)
tsne = TSNE(n_components=2, perplexity=50, n_iter=300)
reduced_observations_tsne = tsne.fit_transform(trajectory_observations)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(reduced_observations_tsne[:, 0], reduced_observations_tsne[:, 1], c=norm(trajectory_rewards), cmap='viridis', marker='o')
plt.title('2D visualization of the original trajectory (t-SNE) with normalized rewards')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.colorbar(scatter, label='Normalized Reward')
plt.savefig('visual/original_trajectory_tsne_visualization_with_normalized_rewards.png')
plt.show()


# ================== Visualization After phi_net (Processed Data) ==================

# PCA Visualization (Processed)
reduced_processed_observations = pca.fit_transform(processed_observations)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(reduced_processed_observations[:, 0], reduced_processed_observations[:, 1], c=norm(trajectory_rewards), cmap='viridis', marker='o')
plt.title('2D visualization of the processed trajectory (PCA) with normalized rewards')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter, label='Normalized Reward')
plt.savefig('visual/processed_trajectory_pca_visualization_with_normalized_rewards.png')
plt.show()

# t-SNE Visualization (Processed)
reduced_processed_observations_tsne = tsne.fit_transform(processed_observations)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(reduced_processed_observations_tsne[:, 0], reduced_processed_observations_tsne[:, 1], c=norm(trajectory_rewards), cmap='viridis', marker='o')
plt.title('2D visualization of the processed trajectory (t-SNE) with normalized rewards')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.colorbar(scatter, label='Normalized Reward')
plt.savefig('visual/processed_trajectory_tsne_visualization_with_normalized_rewards.png')
plt.show()

