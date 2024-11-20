import torch
import argparse
# from dataset import get_dataset, RepeatedDataset
# from get_args import get_git_diff, git_commit
# from NN import other contrastive objectives
# from advance_NN import *
import subprocess
from tqdm import tqdm
from datetime import datetime
import numpy as np
import random
import d4rl
import gym
import wandb
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam 



device = torch.device('cuda:1')

# def weights_init_(m):
#     if isinstance(m, nn.Linear):
#         torch.nn.init.xavier_uniform_(m.weight, gain=1)
#         torch.nn.init.constant_(m.bias, 0)    


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


def get_args():
    parser =  argparse.ArgumentParser()
    parser.add_argument("--seed", help="seed",type=int, default=1234567)
    parser.add_argument("--env_name", help="env_name", type=str, default='halfcheetah-random-v2')
    parser.add_argument("--batch_size", help="BS", type=int, default=4096)
    parser.add_argument("--N", help="N", type=int, default=200)
    parser.add_argument("--eval", help="eval", type=int, default=1)
    parser.add_argument("--type", help="type", type=str, default="pd")
    args = parser.parse_args()
    return args

        
    return model
if __name__ == "__main__":
    args = get_args()    
        
    seed = args.seed
    if args.type == "normal": suffix = ""
    elif args.type == "pd": suffix = "_pd"
    else: raise NotImplementedError("Error!")
    
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) # when using multiple GPUs torch.cuda.manual_seed(seed)
    np.random.seed(seed) 
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
    
    print("evaluation!")
    model = torch.load("model/"+args.env_name+"_contrastive"+str(suffix)+".pt")
    model.to(device)
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]}")

