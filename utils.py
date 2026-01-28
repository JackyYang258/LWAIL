import gym
import numpy as np
import random
import torch
import os
import datetime
import d4rl
from torch import autograd

def log_time():
    # Get the current time
    current_time = datetime.datetime.now()

    # Format the current time
    formatted_time = current_time.strftime("%H:%M:%S")

    # Print the formatted time
    print("Current Time:", formatted_time)
    
def set_seed_everywhere(env: gym.Env, seed=0):
    random.seed(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    # env.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    
def get_normalized_score(env_name, score):
    if env_name in ["Humanoid-v2", 'Humanoid-v3']:
        return score
    ref_min_score = d4rl.infos.REF_MIN_SCORE[env_name]
    ref_max_score = d4rl.infos.REF_MAX_SCORE[env_name]
    return (score - ref_min_score) / (ref_max_score - ref_min_score)

def print_args(args):
    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

def gradient_penalty(netD, real_data, fake_data, l=10):
    batch_size = real_data.size(0)
    alpha = real_data.new_empty((batch_size, 1)).uniform_(0, 1)
    alpha = alpha.expand_as(real_data)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=real_data.new_ones(disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    gradient_penalty = ((gradients_norm - 1) ** 2).mean() * l

    return gradient_penalty