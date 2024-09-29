
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
# icvf_hidden_dims = [11] + [256, 256]
icvf_hidden_dims = [17] + [256, 256]
phi_net = PhiNet(icvf_hidden_dims)
print("phi_net:", phi_net)

# icvf_path = "/home/kaiyan3/siqi/IntentDICE/model/hopper.pt"
icvf_path = "/home/kaiyan3/siqi/IntentDICE/model/halfcheetah.pt"
phi_net.load_state_dict(torch.load(icvf_path))
for param in phi_net.parameters():
    param.requires_grad = False
phi_net.to(device)
print('Using ICVF')

os.makedirs('visual', exist_ok=True)
# env_name = "hopper-expert-v2"
env_name = "halfcheetah-expert-v2"
savedir = "one_expert_trajectory"
file_path = os.path.join(savedir, env_name + ".pkl")

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

# # Plot PCA results before and after using PhiNet, with line segments connecting adjacent points
# plt.figure(figsize=(10, 5))

# # PCA before with lines connecting adjacent points
# plt.subplot(1, 2, 1)
# plt.scatter(pca_obs_before[:, 0], pca_obs_before[:, 1], c='blue', label='Before PhiNet', alpha=0.6)
# plt.plot(pca_obs_before[:, 0], pca_obs_before[:, 1], color='blue', alpha=0.3)  # Connect adjacent points
# plt.title('PCA - Before PhiNet (with line segments)')
# plt.xlabel('PC 1')
# plt.ylabel('PC 2')
# plt.legend()

# # PCA after with lines connecting adjacent points
# plt.subplot(1, 2, 2)
# plt.scatter(pca_obs_after[:, 0], pca_obs_after[:, 1], c='green', label='After PhiNet', alpha=0.6)
# plt.plot(pca_obs_after[:, 0], pca_obs_after[:, 1], color='green', alpha=0.3)  # Connect adjacent points
# plt.title('PCA - After PhiNet (with line segments)')
# plt.xlabel('PC 1')
# plt.ylabel('PC 2')
# plt.legend()

# # 保存为矢量图，确保没有多余的边框
# plt.savefig('visualicvf/pca_comparison_with_lines.svg', format='svg', bbox_inches='tight', dpi=300)
# plt.show()

tsne = TSNE(n_components=2, random_state=10)
tsne_obs_before = tsne.fit_transform(observations)
tsne_obs_after = tsne.fit_transform(obs)

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# t-SNE before with lines connecting adjacent points
axs[0].scatter(tsne_obs_before[:, 0], tsne_obs_before[:, 1], c='blue', label='Before PhiNet', alpha=0.6)
axs[0].plot(tsne_obs_before[:, 0], tsne_obs_before[:, 1], color='blue', alpha=0.3)  # Connect adjacent points
axs[0].set_title('t-SNE of one trajectory state - state space')
axs[0].set_xlabel('Dim 1')
axs[0].set_ylabel('Dim 2')
axs[0].legend()

# t-SNE after with lines connecting adjacent points
axs[1].scatter(tsne_obs_after[:, 0]+15, tsne_obs_after[:, 1], c='green', label='After PhiNet', alpha=0.6)
axs[1].plot(tsne_obs_after[:, 0]+15, tsne_obs_after[:, 1], color='green', alpha=0.3)  # Connect adjacent points
axs[1].set_title('t-SNE of one trajectory state - latent space')
axs[1].set_xlabel('Dim 1')
axs[1].set_ylabel('Dim 2')
axs[1].legend()

# 调整子图之间的间距
plt.subplots_adjust(wspace=0.2, hspace=0.3)

# 保存为PDF格式，确保没有多余的边框
plt.savefig('visualicvf/tsne_comparison_with_lines.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.show()



