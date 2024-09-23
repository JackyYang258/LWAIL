
import os
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pickle

# Assuming you have already defined your device (cpu or cuda)
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

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
icvf_hidden_dims = [17] + [256, 256]
phi_net = PhiNet(icvf_hidden_dims)
print("phi_net:", phi_net)

icvf_path = "/home/kaiyan3/siqi/IntentDICE/model/halfcheetah100k.pt"
phi_net.load_state_dict(torch.load(icvf_path))
for param in phi_net.parameters():
    param.requires_grad = False
phi_net.to(device)
print('Using ICVF')

os.makedirs('visual', exist_ok=True)
env_name = "halfcheetah-expert-v2"
savedir = "one_expert_trajectory"
file_path = os.path.join(savedir, env_name + "_first_episode.pkl")

with open(file_path, 'rb') as f:
    expert_dataset = pickle.load(f)

print("dataset size:", expert_dataset['observations'].shape[0])

# Extract a trajectory
observations = expert_dataset['observations'][:40]
print("observations:", observations.shape)

# Convert the trajectory observations to torch tensors
trajectory_observations_tensor = torch.tensor(observations, dtype=torch.float32).to(device)

print("trajectory_observations_tensor:", trajectory_observations_tensor)
print("observations:", observations)
# Use phi_net to process the observations
with torch.no_grad():
    processed_observations = phi_net(trajectory_observations_tensor)
    if torch.isnan(processed_observations).any():
        print("processed_observations contains NaN")
    if torch.isinf(processed_observations).any():
        print("processed_observations contains Inf")
    obs = processed_observations.cpu().numpy()


# Perform PCA on original and processed observations
pca = PCA(n_components=2)
pca_obs_before = pca.fit_transform(observations)
pca_obs_after = pca.fit_transform(obs)

print("pca_obs_before:", pca_obs_before)
print("pca_obs_after:", pca_obs_after)
# Perform t-SNE on original and processed observations
tsne = TSNE(n_components=2, random_state=10)
tsne_obs_before = tsne.fit_transform(observations)
tsne_obs_after = tsne.fit_transform(obs)
print("tsne_obs_before:", tsne_obs_before)
print("tsne_obs_after:", tsne_obs_after)

# Plot PCA results before and after using PhiNet, with line segments connecting adjacent points
plt.figure(figsize=(10, 5))

# PCA before with lines connecting adjacent points
plt.subplot(1, 2, 1)
plt.scatter(pca_obs_before[:, 0], pca_obs_before[:, 1], c='blue', label='Before PhiNet', alpha=0.6)
plt.plot(pca_obs_before[:, 0], pca_obs_before[:, 1], color='blue', alpha=0.3)  # Connect adjacent points
plt.title('PCA - Before PhiNet (with line segments)')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend()

# PCA after with lines connecting adjacent points
plt.subplot(1, 2, 2)
plt.scatter(pca_obs_after[:, 0], pca_obs_after[:, 1], c='green', label='After PhiNet', alpha=0.6)
plt.plot(pca_obs_after[:, 0], pca_obs_after[:, 1], color='green', alpha=0.3)  # Connect adjacent points
plt.title('PCA - After PhiNet (with line segments)')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend()

plt.savefig('visualicvf/pca_comparison_with_lines.png')
plt.show()

# Plot t-SNE results before and after using PhiNet, with line segments connecting adjacent points
plt.figure(figsize=(10, 5))

# t-SNE before with lines connecting adjacent points
plt.subplot(1, 2, 1)
plt.scatter(tsne_obs_before[:, 0], tsne_obs_before[:, 1], c='blue', label='Before PhiNet', alpha=0.6)
plt.plot(tsne_obs_before[:, 0], tsne_obs_before[:, 1], color='blue', alpha=0.3)  # Connect adjacent points
plt.title('t-SNE - Before PhiNet (with line segments)')
plt.xlabel('Dim 1')
plt.ylabel('Dim 2')
plt.legend()

# t-SNE after with lines connecting adjacent points
plt.subplot(1, 2, 2)
plt.scatter(tsne_obs_after[:, 0], tsne_obs_after[:, 1], c='green', label='After PhiNet', alpha=0.6)
plt.plot(tsne_obs_after[:, 0], tsne_obs_after[:, 1], color='green', alpha=0.3)  # Connect adjacent points
plt.title('t-SNE - After PhiNet (with line segments)')
plt.xlabel('Dim 1')
plt.ylabel('Dim 2')
plt.legend()

plt.savefig('visualicvf/tsne_comparison_with_lines.png')
plt.show()



