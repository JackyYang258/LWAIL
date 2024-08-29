import torch
from tqdm import tqdm

from network import network_weight_matrices, FullyConnectedNet, PhiNet
from ppo import PPO
from td3 import TD3
from utils import time, gradient_penalty
import icecream as ic
import d4rl
import os
import matplotlib.pyplot as plt
import gym
from datetime import datetime
import wandb
import numpy as np

class Agent:
    def __init__(self, state_dim, action_dim, env, expert_buffer, args):
        # Basic information
        self.args = args
        self.only_state = args.only_state
        self.expert_sample = True
        self.update_everystep = args.update_everystep
        max_action = float(env.action_space.high[0])
        self.agent_kind = 'td3'
        if self.agent_kind == 'ppo':
            self.agent = PPO(state_dim, action_dim, self.args.lr_actor, self.args.lr_critic, self.args.gamma, self.args.agent_epoch, self.args.eps_clip, self.args.action_std_init)
        if self.agent_kind == 'td3':
            self.agent = TD3(state_dim, action_dim, self.args.lr_actor, self.args.lr_critic, self.args.agent_epoch, max_action)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.max_action = float(env.action_space.high[0])
        print("action_space:", env.action_space)
        print(f"max_action: {self.max_action}")
        self.env = env

        self.expert_states_buffer = torch.tensor(expert_buffer['observations']).float().to(self.device)
        self.expert_next_states_buffer = torch.tensor(expert_buffer['next_observations']).float().to(self.device)
        self.expert_states = torch.tensor(expert_buffer['observations']).float().to(self.device)
        self.expert_next_states = torch.tensor(expert_buffer['next_observations']).float().to(self.device)
        self.hidden_dims = list(map(int, args.hidden_dim.split(',')))
        torch.set_default_dtype(torch.float32)
        self.filename = "f_net.pth"
        os.makedirs('./log', exist_ok=True)

        # Variable for record
        self.time_step = 0
        self.sum_episodes_reward = 0
        self.sum_episodes_num = 0
        self.i_episode = 0
        self.sample_states = []
        self.sample_next_states = []
        
        # Network initialization
        if self.args.using_icvf:
            icvf_hidden_dims = [state_dim] + [256,256]
            self.phi_net = PhiNet(icvf_hidden_dims)
            print("phi_net:", self.phi_net)
            env_firstname = self.args.env_name.split('-')[0]
            icvf_path = "/scratch/bdaw/kaiyan289/IntentDICE/model/" + env_firstname + ".pt"
            self.phi_net.load_state_dict(torch.load(icvf_path))
            for param in self.phi_net.parameters():
                param.requires_grad = False
            self.phi_net.to(self.device)
            print('Using ICVF')
        else:
            self.phi_net = None
            print('Not using ICVF')

        if self.only_state:
            if self.args.using_icvf:
                self.f_net = FullyConnectedNet(256, self.hidden_dims).to('cuda:0')
            else:
                self.f_net = FullyConnectedNet(state_dim, self.hidden_dims).to('cuda:0')
        else:
            if self.args.using_icvf:
                self.f_net = FullyConnectedNet(256 * 2, self.hidden_dims).to('cuda:0')
            else:
                self.f_net = FullyConnectedNet(state_dim * 2, self.hidden_dims).to('cuda:0')
        print("f_net:", self.f_net)
        # self.f_net = network_weight_matrices(self.f_net, 1)
        self.f_optimizer = torch.optim.Adam(self.f_net.parameters(), self.args.lr_f)

    def train(self):

        while self.time_step <= self.args.max_training_timesteps:
            state = self.env.reset()
            current_ep_reward = 0 # reward for the current episode

            for _ in range(1, self.args.max_ep_len + 1):
                if self.time_step < self.args.start_timesteps:
                    action = self.env.action_space.sample()
                else:
                    action = self.agent.select_action_withrandom(np.array(state))
                next_state, reward, done, _ = self.env.step(action)
                done_bool = float(done)
                self.agent.buffer.add(state, action, next_state, reward, done_bool)
                state = next_state
                
                self.sample_states.append(torch.FloatTensor(state))
                self.sample_next_states.append(torch.FloatTensor(next_state))
                self.time_step += 1
                current_ep_reward += reward

                if self.time_step % self.args.update_timestep == 0:
                    self.sample_states = torch.squeeze(torch.stack(self.sample_states, dim=0)).detach().to(self.device)
                    self.sample_next_states = torch.squeeze(torch.stack(self.sample_next_states, dim=0)).detach().to(self.device)
                    if self.args.using_icvf:
                        self.sample_states = self.phi_net(self.sample_states)
                        self.sample_next_states = self.phi_net(self.sample_next_states)
                    
                    self.f_update()

                    self.get_pseudo_rewards() # Update the buffer with pseudo rewards

                    if not self.update_everystep:
                        self.agent.update() # Update the agent with the pseudo rewards
                    
                    # empty the buffer of f_net update
                    self.sample_states = []
                    self.sample_next_states = []
                    # if self.time_step % 200000 == 0:
                    #     self.generate_heat()

                if self.update_everystep and self.time_step > self.args.update_timestep:
                    self.agent.train()
                    
                if self.time_step % self.args.eval_freq == 0:
                    self.evaluation()
                    self.evaluate_policy()
                
                if done:
                    break

            self.sum_episodes_reward += current_ep_reward
            self.sum_episodes_num += 1
            self.i_episode += 1

        self.save_model()

    def get_pseudo_rewards(self):
        coeff = self.args.reward_coeff

        buffer_state = torch.from_numpy(self.agent.buffer.state).to(self.device).float()
        buffer_next_state = torch.from_numpy(self.agent.buffer.next_state).to(self.device).float()
        if self.args.using_icvf:
            buffer_state = self.phi_net(buffer_state)
            buffer_next_state = self.phi_net(buffer_next_state)

        with torch.no_grad():
            if self.only_state:
                expert_rewards = self.f_net(buffer_state).view(-1)
                expert_rewards = -(expert_rewards - self.f_net(self.expert_states).mean()) * coeff
                expert_rewards = expert_rewards.view(-1, 1).detach().cpu().numpy()
            else:
                expert_rewards = self.f_net(buffer_state, buffer_next_state).view(-1)
                expert_rewards = -(expert_rewards - self.f_net(self.expert_states, self.expert_next_states).mean()) * coeff
                expert_rewards = expert_rewards.view(-1, 1).detach().cpu().numpy()

        self.agent.buffer.rewards = expert_rewards

    def f_update(self):
        if self.expert_sample:
            expert_indices = torch.randint(0, len(self.expert_states_buffer), (len(self.sample_states),))
            self.expert_states = self.expert_states_buffer[expert_indices]
            self.expert_next_states = self.expert_next_states_buffer[expert_indices]
            if self.args.using_icvf:
                self.expert_states = self.phi_net(self.expert_states)
                self.expert_next_states = self.phi_net(self.expert_next_states)
        
        for _ in range(1, self.args.f_epoch + 1):
            if self.only_state:
                loss_f = (torch.mean(self.f_net(self.expert_states)) - 
                        torch.mean(self.f_net(self.sample_states)))
            else:
                loss_f = (torch.mean(self.f_net(self.expert_states, self.expert_next_states)) - 
                    torch.mean(self.f_net(self.sample_states, self.sample_next_states)))

            # Compute the current mean output of f_net using expert states and calculate penalty
            coefficient = 0
            if coefficient != 0:
                if self.only_state:
                    current_mean_f_net = torch.mean(self.f_net(self.expert_states))
                else:
                    current_mean_f_net = torch.mean(self.f_net(self.expert_states, self.expert_next_states))
                penalty_f_value = torch.square(current_mean_f_net - 0)
            
            if self.only_state:
                gradient_penalty_f = gradient_penalty(self.f_net, 
                                                  self.expert_states,
                                                  self.sample_states, l=self.args.alpha)
            else:
                gradient_penalty_f = gradient_penalty(self.f_net, 
                                                    torch.cat((self.expert_states, self.expert_next_states), dim=-1), 
                                                    torch.cat((self.sample_states, self.sample_next_states), dim=-1), l=self.args.alpha)
            coefficient = 0
            if coefficient != 0:
                if self.only_state:
                    current_mean_f_net = torch.mean(self.f_net(self.expert_states))
                else:
                    current_mean_f_net = torch.mean(self.f_net(self.expert_states, self.expert_next_states))
                penalty_f_value = torch.square(current_mean_f_net - 0)
                total_loss_f = loss_f + gradient_penalty_f + coefficient * penalty_f_value
            else:
                total_loss_f = loss_f + gradient_penalty_f

            # Optimize f_net by minimizing total_loss_f
            self.f_net.zero_grad()
            total_loss_f.backward()
            self.f_optimizer.step()
        
        # record f_loss and f_value
        with torch.no_grad():
            if self.only_state:
                loss_f = (torch.mean(self.f_net(self.expert_states)) - torch.mean(self.f_net(self.sample_states)))
            else:
                loss_f = (torch.mean(self.f_net(self.expert_states, self.expert_next_states)) 
                        - torch.mean(self.f_net(self.sample_states, self.sample_next_states)))
            if self.only_state:
                f_value = torch.mean(self.f_net(self.expert_states)).item()
            else:
                f_value = torch.mean(self.f_net(self.expert_states, self.expert_next_states)).item()
        wandb.log({'f_loss': loss_f.item(), 'f_value': f_value}, step=self.time_step)
    
    def evaluation(self):
        avg_reward = round(self.sum_episodes_reward / self.sum_episodes_num, 2)
        normalized_score = d4rl.get_normalized_score(self.args.env_name, avg_reward) * 100
        wandb.log({'train_average_score': avg_reward, 'train_normalized_score': normalized_score}, step=self.time_step)
        self.sum_episodes_reward = 0
        self.sum_episodes_num = 0
    
    def generate_heat(self):
        # Define the input range and sampling density
        x_min, x_max = 0, 5
        y_min, y_max = 0, 5
        density = 0.02

        # Generate x and y coordinates
        x = np.arange(x_min, x_max, density)
        y = np.arange(y_min, y_max, density)
        X, Y = np.meshgrid(x, y)

        # Set f_net to evaluation mode
        self.f_net.eval()

        # Generate input data, first two are xy coordinates, last two are 00
        input_grid = np.c_[X.ravel(), Y.ravel(), np.zeros_like(X.ravel()), np.zeros_like(Y.ravel())]
        input_tensor = torch.tensor(input_grid, dtype=torch.float32).float().to(self.device)

        # Perform forward pass to compute the output
        with torch.no_grad():
            output_tensor = self.get_pseudo_rewards_for_test(input_tensor, input_tensor)

        # Assuming output is a scalar value
        Z = output_tensor.cpu().numpy().reshape(X.shape)

        # Generate heatmap
        plt.figure(figsize=(8, 6))
        plt.contourf(X, Y, Z, levels=100, cmap='viridis')
        plt.colorbar(label='f_net output value')
        plt.title('Heatmap of f_net output')
        plt.xlabel('x')
        plt.ylabel('y')

        # Get the current timestep
        timestep = str(self.time_step)

        # Save the figure with the timestamp in the filename
        plt.savefig(f'log/heatmap_f_net_output_{timestep}.png')
        plt.close()

    def load_model(self):
        """Load the f_net model from the specified filename in the 'model/' directory."""
        path = os.path.join('model', self.filename)
        if os.path.exists(path):
            self.f_net.load_state_dict(torch.load(path))
            print(f"Model loaded from {path}")
        else:
            print(f"No model found at {path}, starting with random weights.")

    def save_model(self):
        """Save the f_net model to the specified filename in the 'model/' directory."""
        os.makedirs('model', exist_ok=True)
        path = os.path.join('model', self.filename)
        torch.save(self.f_net.state_dict(), path)
        print(f"Model saved to {path}")

    def evaluate_policy(self, num_episodes=10):
        env = gym.make(self.args.env_name)
        all_rewards = []

        for episode in range(num_episodes):
            state = env.reset()
            done = False
            episode_rewards = 0

            for step in range(1, self.args.max_ep_len + 1):
                action = self.agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                episode_rewards += reward
                state = next_state
                if done:
                    break
            all_rewards.append(episode_rewards)
        env.close()
        all_rewards = np.array(all_rewards)
        normalized_score = d4rl.get_normalized_score(self.args.env_name, all_rewards.mean()) * 100
        wandb.log({'eval_average_score': all_rewards.mean(), 'eval_normalized_score': normalized_score, 'eval_score_std': all_rewards.std()}, step=self.time_step)

    def get_pseudo_rewards_for_test(self, state, next_state):
        coeff = self.args.reward_coeff
        if self.args.using_icvf:
            state = self.phi_net(state)
            next_state = self.phi_net(next_state)
        with torch.no_grad():
            if self.only_state:
                expert_rewards = self.f_net(next_state).view(-1)
                expert_rewards = -(expert_rewards - self.f_net(self.expert_states).mean()) * coeff
                # expert_rewards = expert_rewards.tolist()
            else:
                expert_rewards = self.f_net(state, next_state).view(-1)
                expert_rewards = -(expert_rewards - self.f_net(self.expert_states, self.expert_next_states).mean()) * coeff
                # expert_rewards = expert_rewards.tolist()
        return expert_rewards