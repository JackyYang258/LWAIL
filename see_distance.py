
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pickle
import numpy as np
import random

class ContrastiveEncoder(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        middle_size = 128
        self.net = nn.Sequential(
            nn.Linear(input_size, middle_size),
            nn.ReLU(),
            nn.Linear(middle_size, middle_size),
            nn.ReLU(),
            nn.Linear(middle_size, output_size))
    
    def forward(self, s):
        return self.net(s)

class Contrastive_PD(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.feature_size = 32
        self.encoder = ContrastiveEncoder(input_size, self.feature_size)
        self.W = torch.nn.Parameter(torch.rand(self.feature_size, self.feature_size)) # note the "distance" is euclidean in embedding space; W does not have to be semi positive-definite
        
    def encode(self, x):
        v = self.encoder(x)
        return v / torch.norm(v, dim=-1, keepdim=True)
        
    def forward(self, s1, s2):
        z1, z2 = self.encode(s1), self.encode(s2)
        # logits = torch.matmul(z1, torch.matmul(self.W, z2.T))
        #print("logits-before:", logits)
        W2 = torch.matmul(F.softplus(self.W), F.softplus(self.W.T))
        logits = torch.matmul(z1, torch.matmul(W2, z2.T))
        # print("logit shape:", logits.shape)
        #print("logits-before:", logits)
        logits -= torch.max(logits, 1)[0][:, None]
        #print("logits-after:", logits)
        return logits 

# Assuming you have already defined your device (cpu or cuda)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

if __name__ == "__main__":
    seed = 0
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) # when using multiple GPUs torch.cuda.manual_seed(seed)
    np.random.seed(seed) 
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
    icvf_hidden_dims = [17] + [256, 256]
    phi_net = PhiNet(icvf_hidden_dims)
    print("phi_net:", phi_net)
    env_fname = "halfcheetah"

    icvf_path = "/home/kaiyan3/siqi/IntentDICE/model/" + env_fname +".pt"
    phi_net.load_state_dict(torch.load(icvf_path))
    for param in phi_net.parameters():
        param.requires_grad = False
    phi_net.to(device)


    os.makedirs('visual', exist_ok=True)
    env_name = env_fname + "-medium-v2"
    savedir = "medium_expert_trajectory"
    file_path = os.path.join(savedir, env_name + ".pkl")

    # model = torch.load("model/"+"halfcheetah-random-v2_contrastive"+"_pd"+".pt")
    # model.to(device)
    # for name, param in model.named_parameters():
    #     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]}")

    with open(file_path, 'rb') as f:
        medium_dataset = pickle.load(f)

    env_name = env_fname + "-expert-v2"
    savedir = "one_expert_trajectory"
    file_path = os.path.join(savedir, env_name + ".pkl")
    with open(file_path, 'rb') as f:
        expert_dataset = pickle.load(f)

    # print("dataset size:", expert_dataset['observations'].shape[0])

    # obeseravtions is the mix of expert_dataset['observations'] and medium_dataset['observations'] first 100
    observations = np.concatenate((expert_dataset['observations'][:100], medium_dataset['observations'][:100]), axis=0)
    print("observations:", observations.shape)
    rewards = np.concatenate((expert_dataset['rewards'][:100], medium_dataset['rewards'][:100]), axis=0)

    rewards_min = rewards.min()
    rewards_max = rewards.max()
    rewards_norm = (rewards - rewards_min) / (rewards_max - rewards_min)
    colors = plt.cm.viridis(rewards_norm)  # 使用viridis颜色映射

    # Convert the trajectory observations to torch tensors
    trajectory_observations_tensor = torch.tensor(observations, dtype=torch.float32).to(device)

    # Use phi_net to process the observations
    with torch.no_grad():
        processed_observations = phi_net(trajectory_observations_tensor)
        print("processed_observations:", processed_observations[0:10])
        if torch.isnan(processed_observations).any():
            print("processed_observations contains NaN")
        if torch.isinf(processed_observations).any():
            print("processed_observations contains Inf")
        obs = processed_observations.cpu().numpy()

    tsne = TSNE(n_components=2, random_state=10)
    tsne_obs_before = tsne.fit_transform(observations)
    tsne_obs_after = tsne.fit_transform(obs)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # t-SNE before with lines connecting adjacent points
    axs[0].scatter(tsne_obs_before[:, 0], tsne_obs_before[:, 1], c=colors[:, 0], label='Before PhiNet', alpha=0.6)
    axs[0].plot(tsne_obs_before[:, 0], tsne_obs_before[:, 1], color='blue', alpha=0.3)
    axs[0].set_title('State Space', fontsize=25)
    axs[0].set_xlabel('Dimension 1', fontsize=20)
    axs[0].set_ylabel('Dimension 2', fontsize=20)
    axs[0].tick_params(axis='both', labelsize=16)  # Adjust tick label size

    # t-SNE after with lines connecting adjacent points
    axs[1].scatter(tsne_obs_after[:, 0] + 15, tsne_obs_after[:, 1], c=colors[:, 0], label='After PhiNet', alpha=0.6)
    axs[1].plot(tsne_obs_after[:, 0] + 15, tsne_obs_after[:, 1], color='green', alpha=0.3) # Connect adjacent points
    axs[1].set_title('Latent Space', fontsize=25)
    axs[1].set_xlabel('Dimension 1', fontsize=20)
    axs[1].set_ylabel('Dimension 2', fontsize=20)
    axs[1].tick_params(axis='both', labelsize=16)  # Adjust tick label size
    # Connect adjacent points

    # t-SNE after with lines connecting adjacent points and color mapping


    # Adjust space between subplots
    plt.subplots_adjust(wspace=0.25, hspace=0.3)
    plt.subplots_adjust(bottom=0.08)

    # Save as PDF with increased DPI and tight bounding box
    plt.savefig('visualicvf/halfcheetah100expert100medium.pdf', format='pdf', bbox_inches='tight', dpi=300)



