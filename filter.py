import gym
import d4rl
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

# ===== 环境配置 =====
env_name = "antmaze-medium-play-v2"
# env_name = "antmaze-medium-diverse-v2"
# env_name = "hopper-expert-v2"
env = gym.make(env_name)

save_dir = "./antmaze_visualizations"
os.makedirs(save_dir, exist_ok=True)

# ===== 加载 offline dataset =====
dataset = d4rl.qlearning_dataset(env)
print(f"Dataset keys: {list(dataset.keys())}")
observations = dataset['observations']
actions = dataset['actions'] # ⭐️ 模仿学习需要 actions
rewards = dataset['rewards']
terminals = dataset['terminals']
# 我们需要 next_observations 来推断 timeouts
next_observations = dataset['next_observations'] 
N = len(observations)

# ===== ⭐️ 关键修改：手动计算 dones (包含 timeouts) =====
dones_float = np.zeros_like(rewards)
for i in range(N - 1):
    is_terminal = dataset['terminals'][i] == 1.0
    is_discontinuous = np.linalg.norm(observations[i + 1] - next_observations[i]) > 1e-6
    if is_terminal or is_discontinuous:
        dones_float[i] = 1
    else:
        dones_float[i] = 0
dones_float[N - 1] = 1
print("Manually computed 'dones_float' (including timeouts) snippet:", dones_float[990:1010])

# ========================================================

# ===== ⭐️ 新增：检查 Reward 统计 =====
unique_rewards = np.unique(rewards)
print(f"\n--- Reward Statistics ---")
print(f"Unique rewards found: {unique_rewards}")
is_0_or_1 = np.all(np.isin(rewards, [0.0, 1.0]))
if is_0_or_1:
    print("Reward check: All rewards are 0.0 or 1.0.")
else:
    print("Reward check: Dataset contains rewards other than 0.0 or 1.0.")
print("---------------------------")

# ========================================================
# ===== ⭐️ 核心修改：过滤专家数据集 =====
# ========================================================

print(f"\n--- Building Trajectory Info & Filtering for Expert Dataset ---")

all_trajectories_info = []
current_episode_reward = 0.0
current_episode_length = 0
current_start_idx = 0

for i in range(N):
    current_episode_reward += rewards[i]
    current_episode_length += 1
    
    if dones_float[i] == 1:
        all_trajectories_info.append({
            'start_idx': current_start_idx,
            'end_idx': i, 
            'length': current_episode_length,
            'return': current_episode_reward
        })
        current_episode_reward = 0.0
        current_episode_length = 0
        current_start_idx = i + 1
        
        if current_start_idx >= N:
            break

# --- 1. 统计所有轨迹 ---
all_lengths = np.array([t['length'] for t in all_trajectories_info])
total_trajs = len(all_trajectories_info)
trajs_len_1 = np.sum(all_lengths == 1)

print(f"Total trajectories found (all types): {total_trajs}")
print(f"Total length=1 trajectories (will be ignored): {trajs_len_1}")
print("Trajectory lengths stats (ALL trajectories):")
print(f"Min: {all_lengths.min()}, Max: {all_lengths.max()}, Mean: {all_lengths.mean():.2f}")


# --- 2. 应用过滤 (length > 1 AND return > 0) ---
expert_trajectories_info = []
for traj_info in all_trajectories_info:
    if traj_info['length'] > 1 and traj_info['return'] > 0:
        expert_trajectories_info.append(traj_info)

print("\n--- Expert Dataset Filtering Results ---")
print(f"High-quality 'expert' trajectories found: {len(expert_trajectories_info)}")

# ⭐️⭐️⭐️ 定义两个新变量，以便在if/else之外访问 ⭐️⭐️⭐️
expert_dataset = {}
expert_trajectories_info_new_indices = []

if len(expert_trajectories_info) == 0:
    print("Warning: No expert trajectories found with criteria (length > 1 AND return > 0)!")
else:
    # --- 3. 构建新的专家数据集字典 ---
    expert_obs_list = []
    expert_act_list = []
    expert_rew_list = []
    expert_next_obs_list = []
    expert_dones_list = []

    for traj_info in expert_trajectories_info:
        start = traj_info['start_idx']
        end = traj_info['end_idx'] + 1
        
        expert_obs_list.append(observations[start:end])
        expert_act_list.append(actions[start:end])
        expert_rew_list.append(rewards[start:end])
        expert_next_obs_list.append(next_observations[start:end])
        expert_dones_list.append(dones_float[start:end])
    
    # 拼接成一个大的 D4RL 格式的字典
    expert_dataset = {
        'observations': np.concatenate(expert_obs_list, axis=0),
        'actions': np.concatenate(expert_act_list, axis=0),
        'rewards': np.concatenate(expert_rew_list, axis=0),
        'next_observations': np.concatenate(expert_next_obs_list, axis=0),
        'dones_float': np.concatenate(expert_dones_list, axis=0)
    }
    
    print(f"New expert dataset created with {len(expert_dataset['observations'])} total transitions.")
    
    # --- 4. ⭐️ 关键：为新的 expert_dataset 创建新的索引元数据 ⭐️ ---
    # 这对于可视化至关重要
    current_idx = 0
    for traj_info in expert_trajectories_info:
        length = traj_info['length']
        expert_trajectories_info_new_indices.append({
            'start_idx': current_idx, # 新的 expert_dataset 中的开始索引
            'end_idx': current_idx + length - 1, # 新的 expert_dataset 中的结束索引
            'length': length,
            'return': traj_info['return']
        })
        current_idx += length
    
    # --- 5. 打印专家数据集的统计信息 ---
    expert_returns = np.array([t['return'] for t in expert_trajectories_info])
    print("\n--- Statistics for *Expert* Dataset ONLY ---")
    print(f"Number of episodes: {len(expert_returns)}")
    print(f"Average Return: {expert_returns.mean():.4f}")
    print(f"Min Return: {expert_returns.min():.4f}")
    print(f"Max Return: {expert_returns.max():.4f}")

print("------------------------------------------------------")

print(f"\n--- Saving Processed Dataset (Pickle format) ---")

# 1. We need 'actions' from the original dataset
actions = dataset['actions']

# 2. Create the dictionary in the exact format your loader expects
#    We include all necessary keys for training.
data_to_save = {
    'observations': observations,
    'actions': actions,
    'next_observations': next_observations,
    'rewards': rewards,
    'terminals': terminals,      # D4RL original 'terminals'
    'dones': dones_float         # Your new 'dones' (with timeouts)
}

# 3. Define save path (using .pkl)
processed_data_dir = "./antmaze_expert_trajectory"
os.makedirs(processed_data_dir, exist_ok=True)
save_path_pkl = os.path.join(processed_data_dir, f"{env_name}.pkl")

print(f"Saving data to: {save_path_pkl}")

# 4. Use pickle.dump() to save the dictionary
#    'wb' means "write binary"
with open(save_path_pkl, 'wb') as f:
    pickle.dump(data_to_save, f)

print("Dataset saved successfully as .pkl file.")
print("------------------------------------------")


# ========================================================
# ===== ⭐️ 可视化部分 (已修改为使用 expert_dataset) ⭐️ =====
# ========================================================

# 仅在 expert_dataset 被成功创建时才执行可视化
if 'observations' in expert_dataset:
    print(f"\n--- Visualizing *Expert* Dataset ---")

    # ===== 1️⃣ 观察 obs[-2:] 方差 =====
    if 'antmaze' in env_name:
        obs_goals = expert_dataset['observations'][:, -2:]
        print("(Expert Viz) obs[-2:] variance:", np.var(obs_goals, axis=0))
    else:
        print("(Expert Viz) Skipping obs[-2:] variance check for non-antmaze env.")

    # ===== 2️⃣ 高 reward 位置 =====
    high_reward_threshold = 0.5
    # 使用 expert_dataset['rewards'] 和 expert_dataset['observations']
    high_reward_idx = expert_dataset['rewards'] >= high_reward_threshold
    high_reward_positions = expert_dataset['observations'][high_reward_idx, 0:2]
    print(f"(Expert Viz) Number of high-reward transitions: {high_reward_positions.shape[0]}")

    # ===== 3️⃣ 用 dones_float 划分轨迹 =====
    # 使用 expert_trajectories_info_new_indices 来找到初始点
    expert_initial_idx = [t['start_idx'] for t in expert_trajectories_info_new_indices]
    initial_positions = expert_dataset['observations'][expert_initial_idx, 0:2]
    print(f"(Expert Viz) Number of trajectories: {len(initial_positions)}")


    # ===== 5️⃣ 采样一些轨迹点可视化 =====
    sample_ratio = 0.05
    num_sample = int(len(expert_dataset['observations']) * sample_ratio)
    # 确保采样数量不超过总数
    if num_sample > 0:
        sample_indices = np.random.choice(len(expert_dataset['observations']), num_sample, replace=False)
        sample_positions = expert_dataset['observations'][sample_indices, 0:2]
    else:
        sample_positions = np.empty((0,2)) # 空数组
    print(f"(Expert Viz) Sampling {num_sample} points...")


    # ===== 6️⃣ 选择长度 >= 100 的轨迹绘制 =====
    MIN_TRAJ_LENGTH = 100
    NUM_TRAJECTORIES_TO_PLOT = 20
    trajectories_to_plot_info = []

    # ⭐️ 使用 expert_trajectories_info_new_indices
    for i in reversed(range(len(expert_trajectories_info_new_indices))): # 从后往前找
        traj_info = expert_trajectories_info_new_indices[i]
        
        if traj_info['length'] >= MIN_TRAJ_LENGTH:
            # 添加一个 traj_idx 键以便在图例中显示
            traj_info['traj_idx'] = i 
            trajectories_to_plot_info.append(traj_info)

        if len(trajectories_to_plot_info) >= NUM_TRAJECTORIES_TO_PLOT:
            break

    print(f"(Expert Viz) Plotting {len(trajectories_to_plot_info)} trajectories with length >= {MIN_TRAJ_LENGTH}.")

    # ===== 7️⃣ 绘图 =====
    plt.figure(figsize=(10,10))
    plt.scatter(sample_positions[:,0], sample_positions[:,1], alpha=0.3, label='sampled positions', s=10)
    plt.scatter(high_reward_positions[:,0], high_reward_positions[:,1], color='red', label='high reward', s=20)
    plt.scatter(initial_positions[:,0], initial_positions[:,1], color='green', marker='x', label='initial points', s=50)

    colors = plt.cm.viridis(np.linspace(0, 1, len(trajectories_to_plot_info)))
    for idx, traj_info in enumerate(trajectories_to_plot_info):
        start_idx = traj_info['start_idx']
        end_idx = traj_info['end_idx']
        
        # ⭐️ 确保从 expert_dataset 中切片
        traj_positions = expert_dataset['observations'][start_idx:end_idx+1, 0:2]

        current_color = colors[idx]
        plt.plot(traj_positions[:,0], traj_positions[:,1], color=current_color, linewidth=2, alpha=0.9,
                 label=f'Traj {traj_info["traj_idx"]} (L:{traj_info["length"]})')

        # 起点
        plt.scatter(traj_positions[0,0], traj_positions[0,1], color='blue', marker='o', s=100, edgecolors='black', zorder=5)
        # 终点
        plt.scatter(traj_positions[-1,0], traj_positions[-1,1], color='magenta', marker='*', s=150, edgecolors='black', zorder=5)

    # 图例
    plt.scatter([], [], color='blue', marker='o', s=100, label='Traj Start', edgecolors='black')
    plt.scatter([], [], color='magenta', marker='*', s=150, label='Traj End', edgecolors='black')

    if 'hopper' in env_name:
        plt.xlabel('torso_x_position (or obs[0])')
        plt.ylabel('torso_y_position (or obs[1])')
    elif 'antmaze' in env_name:
        plt.xlabel('torso_x')
        plt.ylabel('torso_y')
        plt.axis('equal') # Antmaze 必须设置
    else:
        plt.xlabel('obs[0]')
        plt.ylabel('obs[1]')

    # ⭐️ 更新标题
    plt.title(f'{env_name} - EXPERT Trajectories (Return > 0, Length > 1)')
    plt.legend(loc='upper right', bbox_to_anchor=(1.25,1))
    plt.grid(True)


    # ⭐️ 更新保存路径
    save_path = os.path.join(save_dir, f"{env_name}_EXPERT_trajectories.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Expert visualization saved to {save_path}")

else:
    print("\nNo expert data found, skipping visualization.")