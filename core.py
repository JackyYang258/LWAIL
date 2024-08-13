import torch
from tqdm import tqdm

from network import network_weight_matrices, FullyConnectedNet, PhiNet
from ppo import PPO
from utils import time
import icecream as ic
import d4rl
import os
import matplotlib.pyplot as plt
import gym
from datetime import datetime

class Agent:
    def __init__(self, state_dim, action_dim, env, expert_buffer, args):
        # Basic information
        self.args = args
        self.agent = PPO(state_dim, action_dim, self.args.lr_actor, self.args.lr_critic, self.args.gamma, self.args.ppo_epochs, self.args.eps_clip, self.args.action_std_init)
        self.env = env
        self.expert_states = torch.tensor(expert_buffer['observations']).float().to(self.agent.device)
        self.expert_next_states = torch.tensor(expert_buffer['next_observations']).float().to(self.agent.device)
        self.hidden_dims = list(map(int, args.hidden_dim.split(',')))
        torch.set_default_dtype(torch.float32)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Variable for record
        self.time_step = 0
        self.sum_episodes_reward = 0
        self.sum_episodes_num = 0
        self.i_episode = 0
        self.sample_states = []
        self.sample_next_states = []
        
        self.timesteps = []
        self.avg_score = []
        self.normalized_scores = []
        self.f_loss_record = []
        self.time_step_f = []
        
        # Network initialization
        if self.args.using_icvf:
            self.phi_net = PhiNet(icvf_hidden_dims)
            self.phi_net.load_state_dict(torch.load(self.args.icvf_path))
            for param in self.phi_net.parameters():
                param.requires_grad = False
            print('Using ICVF')
        else:
            self.phi_net = None
            print('Not using ICVF')
        self.f_net = FullyConnectedNet(state_dim * 2, self.hidden_dims).to('cuda:0')
        self.f_net = network_weight_matrices(self.f_net, 1)
        self.f_optimizer = torch.optim.Adam(self.f_net.parameters(), self.args.lr_f)
        
        os.makedirs('./log', exist_ok=True)

    def train(self):
        # Retrieve parameters from self.args
        self.sum_episodes_reward = 0
        self.sum_episodes_num = 0

        self.evaluate_policy()

        while self.time_step <= self.args.max_training_timesteps:
            state = self.env.reset()
            current_ep_reward = 0

            for step in range(1, self.args.max_ep_len + 1):
                action = self.agent.select_action(state)
                next_state, reward, done, _ = self.env.step(action)

                self.agent.buffer.next_states.append(torch.tensor(next_state))
                self.agent.buffer.rewards.append(reward)
                self.agent.buffer.is_terminals.append(done)

                self.time_step += 1
                current_ep_reward += reward

                # Update if it's time
                if self.time_step % self.args.update_timestep == 0:
                    self.sample_states = torch.squeeze(torch.stack(self.agent.buffer.states, dim=0)).detach().to(self.agent.device)
                    self.sample_next_states = torch.squeeze(torch.stack(self.agent.buffer.states, dim=0)).detach().to(self.agent.device)

                    self.f_update()

                    if self.time_step > 150000:
                        print("before", self.agent.buffer.rewards[500:503])

                    self.agent.buffer.rewards = self.get_pseudo_rewards()

                    if self.time_step > 150000:
                        print("after", self.agent.buffer.rewards[500:503])

                    if self.time_step > 250000:
                        self.print_pertubarion_results_for_test()

                    self.agent.update()
                    self.agent.buffer.clear()

                if self.time_step % self.args.action_std_decay_frequency == 0:
                    self.agent.decay_action_std(self.args.action_std_decay_rate, self.args.min_action_std)

                if self.time_step % self.args.eval_freq == 0:
                    self.evaluation()

                if done:
                    break

            self.sum_episodes_reward += current_ep_reward
            self.sum_episodes_num += 1

            self.i_episode += 1

        # Plotting and saving the results
        self.save_results()

    def evaluate_policy(self, goal_state=None, num_episodes=10):
        env = gym.make(self.args.env_name)
        all_states = []
        all_rewards = []

        for episode in range(num_episodes):
            state = env.reset()
            done = False
            episode_states = []
            episode_rewards = []

            for step in range(1, 1000 + 1):
                action = self.agent.select_action_eval(state)
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
            print("sum of rewards:", sum(episode_rewards))
        print("mean rewards of all episodes:", sum(map(sum, all_rewards)) / num_episodes)

    def get_pseudo_rewards(self):
        coeff = 0.2
        with torch.no_grad():
            expert_rewards = self.f_net(self.sample_states, self.sample_next_states).view(-1)
            expert_rewards = -(expert_rewards - self.f_net(self.expert_states, self.expert_next_states).mean()) * coeff
            expert_rewards = expert_rewards.tolist()
        return expert_rewards
    
    def get_pseudo_rewards_for_test(self, state, next_state):
        coeff = 0.2
        with torch.no_grad():
            expert_rewards = self.f_net(state, next_state).view(-1)
            expert_rewards = -(expert_rewards - self.f_net(self.expert_states, self.expert_next_states).mean()) * coeff
            expert_rewards = expert_rewards.tolist()
        return expert_rewards

    def f_update(self):
        previous_loss_f = float('inf')
        converged = False
        first_loss_f = 0
        
        for f_step in range(1, self.args.f_epoch + 1):
            if self.args.using_icvf:
                loss_f = (torch.mean(self.f_net(self.phi_net(self.expert_states), self.phi_net(self.expert_next_states))) - torch.mean(self.f_net(self.phi_net(self.sample_states), self.phi_net(self.sample_next_states))))
            else:
                loss_f = (torch.mean(self.f_net(self.expert_states, self.expert_next_states)) - torch.mean(self.f_net(self.sample_states, self.sample_next_states)))
            
            if f_step == 1 and self.f_loss_record != []:
                print(f'f_loss_difference after update ppo: {loss_f.item() - self.f_loss_record[-1]}')
                first_loss_f = loss_f.item()
                
                

            if converged and abs(previous_loss_f - loss_f) < 1e-3:
                print(f'Converged at step {f_step}')
                break

            if abs(previous_loss_f - loss_f) < 1e-5:
                converged = True
                print("1")
                break

            previous_loss_f = loss_f.item()

            # Optimize f_net by minimizing loss_f
            self.f_net.zero_grad()
            loss_f.backward()
            self.f_optimizer.step()

            self.f_net = network_weight_matrices(self.f_net, 1)
        
        with torch.no_grad():
            if self.args.using_icvf:
                loss_f = (torch.mean(self.f_net(self.phi_net(self.expert_states), self.phi_net(self.expert_next_states))) - torch.mean(self.f_net(self.phi_net(self.sample_states), self.phi_net(self.sample_next_states))))
            else:
                loss_f= (torch.mean(self.f_net(self.expert_states, self.expert_next_states)) - torch.mean(self.f_net(self.sample_states, self.sample_next_states)))

        print(f'f_loss: {loss_f.item()}')
        print(f'f_loss_difference after update f: {loss_f.item() - first_loss_f}')
        self.f_loss_record.append(loss_f.item())
        self.time_step_f.append(self.time_step)
        
    def print_pertubarion_results_for_test(self):
        # Initial state
        initial_state = torch.tensor([0.5, 0.5, 0.0, 0.0]).to(self.agent.device)

        # Small perturbations, including zero perturbation
        perturbations = torch.tensor([
            [-0.01, 0.0, 0.0, 0.0],   # Left
            [0.0, 0.01, 0.0, 0.0],    # Up
            [0.01, 0.0, 0.0, 0.0],    # Right
            [0.0, -0.01, 0.0, 0.0],   # Down
            [0.01, 0.01, 0.0, 0.0],   # Up-right
            [-0.01, -0.01, 0.0, 0.0], # Down-left
            [0.01, -0.01, 0.0, 0.0],  # Down-right
            [-0.01, 0.01, 0.0, 0.0],  # Up-left
            [0.0, 0.0, 0.0, 0.0]      # Center
        ]).to(self.agent.device)

        # Get f_net results and store them
        results = []
        for perturbation in perturbations:
            next_state = initial_state + perturbation
            output_reward = self.get_pseudo_rewards_for_test(initial_state, next_state)[0]
            results.append(output_reward)

        # Print results as a 3x3 matrix
        matrix = [
            results[7], results[1], results[4],
            results[0], results[8], results[2],
            results[5], results[3], results[6]
        ]

        print("rewards")
        print("[")
        for i in range(3):
            print(f"  {matrix[i*3:i*3+3]}")
        print("]")
    
    def evaluation(self):
        avg_reward = round(self.sum_episodes_reward / self.sum_episodes_num, 2)
        normalized_score = d4rl.get_normalized_score(self.args.env_name, avg_reward)
        print(f"Episode : {self.i_episode} \t\t Timestep : {self.time_step} \t\t Average Score : {avg_reward}")
        print(f"Episode : {self.i_episode} \t\t Timestep : {self.time_step} \t\t Normalized Average Score : {normalized_score}")

        self.timesteps.append(self.time_step)
        self.avg_score.append(avg_reward)
        self.normalized_scores.append(normalized_score)
        self.sum_episodes_reward = 0
        self.sum_episodes_num = 0
        
    def save_results(self):
        # Get current time for file naming
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Plot and save average score
        plt.figure(figsize=(12, 6))
        plt.plot(self.timesteps, self.avg_score, label='Average Score')
        plt.xlabel('Timesteps')
        plt.ylabel('Average Score')
        plt.legend()
        plt.title('Average Score vs Timesteps')
        plt.savefig(f'./log/average_score_vs_timesteps_{current_time}.png')
        plt.show()

        # Plot and save normalized average score
        plt.figure(figsize=(12, 6))
        plt.plot(self.timesteps, self.normalized_scores, label='Normalized Average Score')
        plt.xlabel('Timesteps')
        plt.ylabel('Normalized Average Score')
        plt.legend()
        plt.title('Normalized Average Score vs Timesteps')
        plt.savefig(f'./log/normalized_average_score_vs_timesteps_{current_time}.png')
        plt.show()

        # Plot and save f-loss
        plt.figure(figsize=(12, 6))
        plt.plot(self.time_step_f, self.f_loss_record, label='f-loss')
        plt.xlabel('Timesteps')
        plt.ylabel('f-loss')
        plt.legend()
        plt.title('f-loss vs Timesteps')
        plt.savefig(f'./log/f_loss_{current_time}.png')
        plt.show()
