import torch
from tqdm import tqdm

from network import network_weight_matrices
from ppo import PPO
from utils import time
import icecream as ic
import d4rl
import os
import matplotlib.pyplot as plt
import gym

using_ICVF = False

def train(expert_buffer, f_net, phi, env, seed, max_ep_len, max_training_timesteps, update_timestep, f_epoch, lr_f, action_std_decay_frequency, action_std_decay_rate, min_action_std, state_dim, action_dim, lr_actor, lr_critic, gamma, ppo_epochs, eps_clip, action_std_init=0.6):
    # def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std_init=0.6):
    time()
    os.makedirs('./log', exist_ok=True)
    
    agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, ppo_epochs, eps_clip, action_std_init)
    
    f_optimizer = torch.optim.Adam(f_net.parameters(), lr=lr_f)
    
    eval_freq = 10000
    time_step = 0
    i_episode = 0
    
    timesteps = []
    avg_score = []
    normalized_scores = []
    f_loss_record = []
    time_step_f = []
    
    print_running_reward = 0
    print_running_episodes = 0
    
    while time_step <= max_training_timesteps:
        
        state = env.reset()
        current_ep_reward = 0
        
        for step in range(1, max_ep_len + 1):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            agent.buffer.next_states.append(torch.tensor(next_state))
            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)
            
            time_step += 1
            current_ep_reward += reward
            
            # update if its time
            if time_step % update_timestep == 0:
                
                s1 = torch.tensor(expert_buffer['observations']).float().to(agent.device)
                s1_prime = torch.tensor(expert_buffer['next_observations']).float().to(agent.device)
                s2 = torch.squeeze(torch.stack(agent.buffer.states, dim=0)).detach().to(agent.device)
                s2_prime = torch.squeeze(torch.stack(agent.buffer.states, dim=0)).detach().to(agent.device)
                
                
                # print loss_f after ppo update
                if using_ICVF:
                    loss_f_after_ppo = (torch.mean(f_net(phi(s1), phi(s1_prime))) - torch.mean(f_net(phi(s2), phi(s2_prime))))
                else:
                    loss_f_after_ppo = (torch.mean(f_net(s1, s1_prime)) - torch.mean(f_net(s2, s2_prime)))
                if f_loss_record != []:
                    print(f'f_loss_difference after update ppo: {loss_f_after_ppo.item() - f_loss_record[-1]}')

                previous_loss_f = float('inf')
                converged = False
                for f_step in range(1, f_epoch + 1):
                    # Calculate the loss
                    
                    if using_ICVF:
                        loss_f = (torch.mean(f_net(phi(s1), phi(s1_prime))) - torch.mean(f_net(phi(s2), phi(s2_prime))))
                    else:
                        loss_f = (torch.mean(f_net(s1, s1_prime)) - torch.mean(f_net(s2, s2_prime)))
                    
                    if converged and abs(previous_loss_f - loss_f) < 1e-3:
                        print(f'Converged at step {f_step}')
                        break
                    else:
                        converged = False
                    
                    if abs(previous_loss_f - loss_f) < 1e-5:
                        converged = True
                        print("1")
                        break
                        
                        
                    previous_loss_f = loss_f.item()
                    # Optimize f_net by minimizing loss_f
                    f_net.zero_grad()
                    loss_f.backward()
                    f_optimizer.step()
                    
                    f_net = network_weight_matrices(f_net, 1)
                    
                # evluate the f-loss
                if using_ICVF:
                    loss_f = (torch.mean(f_net(phi(s1), phi(s1_prime))) - torch.mean(f_net(phi(s2), phi(s2_prime))))
                else:
                    loss_f = (torch.mean(f_net(s1, s1_prime)) - torch.mean(f_net(s2, s2_prime)))
                print(f'f_loss: {loss_f.item()}')
                f_loss_record.append(loss_f.item())
                time_step_f.append(time_step)
                                
                # agent.buffer.rewards = f_net(s,s')
                tensor_states = torch.stack(agent.buffer.states).to(agent.device).float()
                tensor_next_states = torch.stack(agent.buffer.next_states).to(agent.device).float()
                
                if time_step > 150000:
                    print("before",agent.buffer.rewards[500:510])
                agent.buffer.rewards = (-f_net(tensor_states, tensor_next_states)).view(-1).tolist()
                if time_step > 150000:
                    print("after",agent.buffer.rewards[500:510])
                if time_step > 250000:
                    # 定义初始状态
                    initial_state = torch.tensor([0.5, 0.5, 0.0, 0.0])  # 位置 (0.5, 0.5) 速度 (0, 0)

                    # 定义微小的扰动，包括零扰动
                    perturbations = torch.tensor([
                        [-0.01, 0.0, 0.0, 0.0],   # 左
                        [0.0, 0.01, 0.0, 0.0],    # 上
                        [0.01, 0.0, 0.0, 0.0],    # 右
                        [0.0, -0.01, 0.0, 0.0],   # 下
                        [0.01, 0.01, 0.0, 0.0],   # 右上
                        [-0.01, -0.01, 0.0, 0.0], # 左下
                        [0.01, -0.01, 0.0, 0.0],  # 右下
                        [-0.01, 0.01, 0.0, 0.0],  # 左上
                        [0.0, 0.0, 0.0, 0.0]      # 中心
                    ])

                    # 获取 f_net 的结果并存储
                    results = []
                    for perturbation in perturbations:
                        next_state = initial_state + perturbation
                        output_state = -f_net(initial_state.unsqueeze(0).to(agent.device), next_state.unsqueeze(0).to(agent.device)).item()
                        results.append(output_state)

                    # 打印结果为 3x3 矩阵格式
                    matrix = [
                        results[7], results[1], results[4],
                        results[0], results[8], results[2],
                        results[5], results[3], results[6]
                    ]

                    print("[")
                    for i in range(3):
                        print(f"  {matrix[i*3:i*3+3]}")
                    print("]")
                    
                agent.update()
                agent.buffer.clear()
                
            if time_step % action_std_decay_frequency == 0:
                agent.decay_action_std(action_std_decay_rate, min_action_std)
            
            if time_step % eval_freq == 0:
                avg_reward = round(print_running_reward / print_running_episodes, 2)
                normalized_score = d4rl.get_normalized_score(env.spec.id, avg_reward)
                print("Episode : {} \t\t Timestep : {} \t\t Average Score : {}".format(i_episode, time_step, avg_reward))
                print("Episode : {} \t\t Timestep : {} \t\t Normalized Average Score : {}".format(i_episode, time_step, normalized_score))
                
                timesteps.append(time_step)
                avg_score.append(avg_reward)
                normalized_scores.append(normalized_score)
                print_running_reward = 0
                print_running_episodes = 0
                
            if done:
                print(f"Episode {i_episode} finished after {step} timesteps")
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        # log_running_reward += current_ep_reward
        # log_running_episodes += 1

        i_episode += 1
        
    
    def evaluate_policy(agent, env_name, goal_state=None, num_episodes=3):
        env = gym.make(env_name)
        all_states = []
        all_rewards = []

        for episode in range(num_episodes):
            state = env.reset()
            done = False
            episode_states = []
            episode_rewards = []

            for step in range(1, 1000 + 1):
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                episode_states.append(state)
                episode_rewards.append(reward)
                state = next_state
                if done:
                    break

            all_states.append(episode_states)
            all_rewards.append(episode_rewards)

        env.close()
        print("Visited states and distances to goal in each episode:")
        for i, (episode_states, episode_rewards) in enumerate(zip(all_states, all_rewards)):
            print(f"Episode {i+1}:")
            for state, distance in zip(episode_states, episode_rewards):
                print(f"State: {state}, Reward: {distance}")
    # evaluate_policy(agent, env.spec.id)

    from datetime import datetime

    # 获取当前时间并格式化为字符串
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 绘制并保存平均分数图像
    plt.figure(figsize=(12, 6))
    plt.plot(timesteps, avg_score, label='Average Score')
    plt.xlabel('Timesteps')
    plt.ylabel('Average Score')
    plt.legend()
    plt.title('Average Score vs Timesteps')
    plt.savefig(f'./log/average_score_vs_timesteps_{current_time}.png')
    plt.show()

    # 绘制并保存标准化平均分数图像
    plt.figure(figsize=(12, 6))
    plt.plot(timesteps, normalized_scores, label='Normalized Average Score')
    plt.xlabel('Timesteps')
    plt.ylabel('Normalized Average Score')
    plt.legend()
    plt.title('Normalized Average Score vs Timesteps')
    plt.savefig(f'./log/normalized_average_score_vs_timesteps_{current_time}.png')
    plt.show()

    # 绘制并保存f-loss图像
    plt.figure(figsize=(12, 6))
    plt.plot(time_step_f, f_loss_record, label='f-loss')
    plt.xlabel('Timesteps')
    plt.ylabel('f-loss')
    plt.legend()
    plt.title('f-loss vs Timesteps')
    plt.savefig(f'./log/f_loss_{current_time}.png')
    plt.show()

