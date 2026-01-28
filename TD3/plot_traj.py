import numpy as np
import torch
import gym
import matplotlib.pyplot as plt
import argparse
import os

import TD3
import OurDDPG
import DDPG
from main import FixedStartWrapper  # 导入你的 Wrapper

def eval_and_collect_trajectory(policy, env_name, file_name, eval_episodes=5):
    # 创建环境
    env = gym.make(env_name)
    env = FixedStartWrapper(env, fixed_start=[1.0, 1.0, 0.0, 0.0])
    
    # 获取 state/action 空间信息
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    # 判断 policy 类型
    kwargs = {"state_dim": state_dim, "action_dim": action_dim, "max_action": max_action}
    if isinstance(policy, str):
        if policy == "TD3":
            policy = TD3.TD3(**kwargs)
        elif policy == "OurDDPG":
            policy = OurDDPG.DDPG(**kwargs)
        elif policy == "DDPG":
            policy = DDPG.DDPG(**kwargs)
    
    # Load model
    policy.load(f"./models/{file_name}")
    
    all_trajectories = []
    
    for ep in range(eval_episodes):
        state = env.reset()
        done = False
        trajectory = {"states": [], "actions": [], "rewards": []}
        while not done:
            action = policy.select_action(np.array(state))
            next_state, reward, done, _ = env.step(action)
            
            trajectory["states"].append(state.copy())
            trajectory["actions"].append(action.copy())
            trajectory["rewards"].append(reward)
            
            state = next_state
        all_trajectories.append(trajectory)
    
    return all_trajectories

def plot_trajectory(trajectories):
    plt.figure(figsize=(8,6))
    for i, traj in enumerate(trajectories):
        states = np.array(traj["states"])
        plt.plot(states[:,0], states[:,1], marker='o', label=f'Episode {i+1}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('State Trajectories')
    plt.legend()
    plt.grid(True)
    plt.savefig('trajectories.png')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_type", default="TD3", help="Policy type: TD3, DDPG, OurDDPG")
    parser.add_argument("--env", default="maze2d-large-dense-v1", help="Environment name")
    parser.add_argument("--episodes", default=3, type=int, help="Number of episodes to plot")
    parser.add_argument("--policy", default="TD3")
    parser.add_argument("--seed", default=0, type=int)  
    args = parser.parse_args()
    file_name = f"{args.policy}_{args.env}_{args.seed}"
    trajectories = eval_and_collect_trajectory(args.policy_type, args.env, file_name, args.episodes)
    plot_trajectory(trajectories)
