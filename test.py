import d4rl
import gym
import numpy as np
import os

env = gym.make("halfcheetah-medium-expert-v2")
dataset = d4rl.qlearning_dataset(env)
        
limit = int(1e5)
dataset = {k: v[:limit] for k, v in dataset.items()}
