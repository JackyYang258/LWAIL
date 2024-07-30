import gym
import numpy as np
import random
import torch
import os

def set_seed_everywhere(env: gym.Env, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
def select_stochastic_action(policy_net, state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy_net(state)
    action = np.random.choice(len(probs.squeeze()), p=probs.squeeze().detach().numpy())
    return action, probs[:, action].item()