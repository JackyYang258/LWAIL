import gym
import numpy as np
import random
import torch
import os
import datetime
import d4rl
from torch import autograd

def time():
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
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    
def get_normalized_score(env_name, score):
    ref_min_score = d4rl.infos.REF_MIN_SCORE[env_name]
    ref_max_score = d4rl.infos.REF_MAX_SCORE[env_name]
    return (score - ref_min_score) / (ref_max_score - ref_min_score)

def print_args(args):
    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

def gradient_penalty(netD, real_data, fake_data, l=10):
    # batch_size = real_data.size(0)
    # alpha = real_data.new_empty((batch_size, 1)).uniform_(0, 1)
    # alpha = alpha.expand_as(real_data)

    # interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    # interpolates = autograd.Variable(interpolates, requires_grad=True)

    # disc_interpolates = netD(interpolates)

    # gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
    #                           grad_outputs=real_data.new_ones(disc_interpolates.size()),
    #                           create_graph=True, retain_graph=True, only_inputs=True)[0]

    # gradients = gradients.view(gradients.size(0), -1)
    # gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    # gradient_penalty = ((gradients_norm - 1) ** 2).mean() * l

    # return gradient_penalty

    # Determine the minimum batch size between real and fake data
    min_batch_size = min(real_data.size(0), fake_data.size(0))

    # Randomly sample min_batch_size samples from real_data and fake_data
    real_indices = torch.randperm(real_data.size(0))[:min_batch_size]
    fake_indices = torch.randperm(fake_data.size(0))[:min_batch_size]

    real_data_sample = real_data[real_indices]
    fake_data_sample = fake_data[fake_indices]

    # Generate alpha of appropriate size
    alpha = real_data_sample.new_empty((min_batch_size, 1)).uniform_(0, 1)
    alpha = alpha.expand_as(real_data_sample)

    # Create interpolates
    interpolates = alpha * real_data_sample + ((1 - alpha) * fake_data_sample)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    # Get discriminator output for interpolates
    disc_interpolates = netD(interpolates)

    # Compute gradients of the discriminator output with respect to interpolates
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(real_data.device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    # Calculate gradient norm and gradient penalty
    gradients = gradients.view(min_batch_size, -1)
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    gradient_penalty = ((gradients_norm - 1) ** 2).mean() * l

    return gradient_penalty