import d4rl
import gym
import numpy as np
import os

# 创建存储数据的目录
save_dir = "/projects/bdaw/kaiyan289/IntentDICE/d4rl_datasets"
os.makedirs(save_dir, exist_ok=True)

# 定义任务名称
tasks = [
    "halfcheetah-random-v2", "halfcheetah-medium-v2", "halfcheetah-expert-v2",
    "halfcheetah-medium-replay-v2", "halfcheetah-medium-expert-v2",
    "walker2d-random-v2", "walker2d-medium-v2", "walker2d-expert-v2",
    "walker2d-medium-replay-v2", "walker2d-medium-expert-v2",
    "hopper-random-v2", "hopper-medium-v2", "hopper-expert-v2",
    "hopper-medium-replay-v2", "hopper-medium-expert-v2",
    "ant-random-v2", "ant-medium-v2", "ant-expert-v2",
    "ant-medium-replay-v2", "ant-medium-expert-v2"
]


# 遍历任务并保存数据
for task in tasks:
    print(f"Loading dataset for task: {task}")
    env = gym.make(task)
    dataset = env.get_dataset()

    # 提取并保存数据
    save_path = os.path.join(save_dir, f"{task}.npz")
    np.savez(save_path,
             observations=dataset['observations'],
             actions=dataset['actions'],
             rewards=dataset['rewards'],
             terminals=dataset['terminals'],
             next_observations=dataset.get('next_observations', None))
    
    print(f"Dataset saved at: {save_path}")

print("All datasets have been saved.")
