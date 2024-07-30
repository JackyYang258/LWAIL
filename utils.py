import gym
import numpy as np
import random
import torch
import os
import datetime

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
