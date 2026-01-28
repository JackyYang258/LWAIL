import torch
from tqdm import tqdm

from network import FullyConnectedNet, PhiNet
from td3 import TD3
from ddpg import DDPG
from utils import gradient_penalty, get_normalized_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import matplotlib.pyplot as plt
import gymnasium as gym
import shimmy
from datetime import datetime
import wandb
import numpy as np
import minari

class Agent:
    def __init__(self, state_dim, action_dim, env, expert_buffer, args):
        # Basic information
        self.args = args
        self.device = torch.device(self.args.cuda if torch.cuda.is_available() else 'cpu')
        self.using_icvf = args.using_icvf
        self.state_action = args.state_action
        self.expert_sample = True
        self.update_everystep = args.update_everystep
        self.agent_kind = args.downstream
        if self.agent_kind == 'td3':
            self.agent = TD3(state_dim, action_dim, self.args.lr_actor, self.args.lr_critic, self.device, self.args.curl)
        if self.agent_kind == "ddpg":
            self.agent = DDPG(state_dim, action_dim, self.args.lr_actor, self.args.lr_critic, self.device)
        self.max_action = float(env.action_space.high[0])
        print("action_space:", env.action_space)
        print(f"max_action: {self.max_action}")
        if 'dm_control' in self.args.env_name:
            self.env = gym.make(self.args.env_name)
            self.env = gym.wrappers.FlattenObservation(self.env)
        else:
            dataset = minari.load_dataset('mujoco/humanoid/expert-v0')
            self.env = dataset.recover_environment()
        (state, info), done = self.env.reset(), False

        self.dataset = expert_buffer
        self.expert_states_buffer = torch.tensor(expert_buffer['observations']).float().to(self.device)
        self.expert_next_states_buffer = torch.tensor(expert_buffer['next_observations']).float().to(self.device)
        self.expert_actions_buffer = torch.tensor(expert_buffer['actions']).float().to(self.device) 
        self.expert_states_buffer = self.expand_and_fill(self.expert_states_buffer, self.args.update_timestep)
        self.expert_next_states_buffer = self.expand_and_fill(self.expert_next_states_buffer, self.args.update_timestep)
        self.expert_actions_buffer = self.expand_and_fill(self.expert_actions_buffer, self.args.update_timestep)
        
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
        self.sample_actions = []
        
        # Network initialization
        if self.args.using_icvf:
            icvf_hidden_dims = [state_dim] + [256,256]
            self.phi_net = PhiNet(icvf_hidden_dims)
            print("phi_net:", self.phi_net)
            if 'dm_control' in self.args.env_name:
                env_firstname = (self.args.env_name.split('/')[1]).split('-')[0]
            else:
                env_firstname = 'mujocohumanoid'
            model_dir = os.path.join(os.getcwd(), "model")
            icvf_path = os.path.join(model_dir, f"{env_firstname}.pt")
            self.phi_net.load_state_dict(torch.load(icvf_path, weights_only=False))
            for param in self.phi_net.parameters():
                param.requires_grad = False
            self.phi_net.to(self.device)
            print('Using ICVF from', icvf_path)
        else:
            self.phi_net = None
            print('Not using ICVF')
        if self.args.using_pwdice:
            model = torch.load("model/"+self.args.env_name.split('-')[0]+"-random-v2_contrastive"+"_pd"+".pt")
            model.to(self.device)
            self.f_net = model.encoder.float()
            for name, param in self.f_net.named_parameters():
                print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]}")
        if self.args.state_action:
            if self.args.using_icvf:
                self.f_net = FullyConnectedNet(256 + action_dim, self.hidden_dims).to(self.device)
            else:
                self.f_net = FullyConnectedNet(state_dim + action_dim, self.hidden_dims).to(self.device)
        else:
            if self.args.using_icvf:
                self.f_net = FullyConnectedNet(256 * 2, self.hidden_dims).to(self.device)
            else:
                self.f_net = FullyConnectedNet(state_dim * 2, self.hidden_dims).to(self.device)
        print("f_net:", self.f_net)
        self.f_optimizer = torch.optim.Adam(self.f_net.parameters(), self.args.lr_f)

    def train(self):
        self.pretrain()
        while self.time_step <= self.args.max_training_timesteps:
            (state, info), done = self.env.reset(), False
            current_ep_reward = 0
            for _ in range(1, self.args.max_ep_len + 1):
                if self.agent_kind == 'td3' or self.agent_kind == 'ddpg':
                    if self.time_step < self.args.start_timesteps:
                        action = self.env.action_space.sample()
                    else:
                        action = self.agent.select_action_withrandom(state)
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    done = float(truncated or terminated)
                    if self.time_step > -1:
                        state_tensor = torch.FloatTensor(state).to(self.device)
                        next_state_tensor = torch.FloatTensor(next_state).to(self.device)
                        if self.state_action:
                            action_tensor = torch.FloatTensor(action).to(self.device)
                            self.sample_actions.append(action_tensor)
                        self.sample_states.append(state_tensor)
                        self.sample_next_states.append(next_state_tensor)
                        if self.args.using_icvf:
                            state_tensor = self.phi_net(state_tensor)
                            next_state_tensor = self.phi_net(next_state_tensor)
                        if self.args.minus:
                            next_state_tensor = next_state_tensor - state_tensor
                        if self.state_action:
                            fake_reward = -self.f_net(state_tensor, action_tensor)
                        else:
                            fake_reward = -self.f_net(state_tensor, next_state_tensor)
                        fake_reward = torch.sigmoid(fake_reward).detach().cpu().item()
                        
                    self.agent.buffer.add(state, action, next_state, fake_reward, float(done))
                    state = next_state
                elif self.agent_kind == 'onlytd3':
                    if self.time_step < self.args.start_timesteps:
                        action = self.env.action_space.sample()
                    else:
                        action = self.agent.select_action_withrandom(state)
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    done = float(truncated or terminated)
                    # reward = sigmoid(reward)
                    reward = torch.sigmoid(torch.tensor(reward)).item()
                    self.agent.buffer.add(state, action, next_state, reward, float(done))
                    state = next_state
                
                self.time_step += 1
                current_ep_reward += reward
                # print("time_step:", self.time_step)
                if self.time_step % self.args.update_timestep == 0:
                    
                    if self.agent_kind == 'td3' or self.agent_kind == 'ddpg':
                        self.f_update()
                        self.get_pseudo_rewards()
                        
                    # self.get_pseudo_rewards() # Update the buffer with pseudo rewards
                    # if self.agent_kind == 'ppo':
                    #     self.agent.update() # Update the agent with the pseudo rewards

                if self.time_step > self.args.update_timestep:
                    self.agent.train()
                    
                if self.time_step % self.args.eval_freq == 0:
                    self.evaluation()
                    self.evaluate_policy()
                
                if done:
                    break

            self.sum_episodes_reward += current_ep_reward
            self.sum_episodes_num += 1
            self.i_episode += 1
        self.env.close()

    def get_pseudo_rewards(self):
        
        # self.min_ref, self.max_ref = self.return_range()

        buffer_state = torch.FloatTensor(self.agent.buffer.state).to(self.device)
        buffer_next_state = torch.FloatTensor(self.agent.buffer.next_state).to(self.device)
        buffer_action = torch.FloatTensor(self.agent.buffer.action).to(self.device)
        if self.args.using_icvf:
            buffer_state = self.phi_net(buffer_state)
            buffer_next_state = self.phi_net(buffer_next_state)
        if self.args.minus:
            buffer_next_state = buffer_next_state - buffer_state

        if self.state_action:
            fake_reward = -self.f_net(buffer_state, buffer_action).view(-1)
        else:
            fake_reward = -self.f_net(buffer_state, buffer_next_state).view(-1)
        fake_reward = torch.sigmoid(fake_reward)
        
        fake_reward = fake_reward.detach().cpu().numpy().reshape(-1, 1)
            
        self.agent.buffer.reward = fake_reward

    def f_update(self):
        self.sample_states = torch.squeeze(torch.stack(self.sample_states, dim=0)).detach().to(self.device)
        self.sample_next_states = torch.squeeze(torch.stack(self.sample_next_states, dim=0)).detach().to(self.device)
        if self.state_action:
            self.sample_actions = torch.squeeze(torch.stack(self.sample_actions, dim=0)).detach().to(self.device)

        assert len(self.sample_states) == self.args.update_timestep
        expert_indices = torch.randint(0, len(self.expert_states_buffer), (len(self.sample_states),))
        self.expert_states = self.expert_states_buffer[expert_indices].clone()
        self.expert_next_states = self.expert_next_states_buffer[expert_indices].clone()
        if self.state_action:
            self.expert_actions = self.expert_actions_buffer[expert_indices].clone()
        if self.args.using_icvf:
            self.expert_states = self.phi_net(self.expert_states)
            self.expert_next_states = self.phi_net(self.expert_next_states)
            self.sample_states = self.phi_net(self.sample_states)
            self.sample_next_states = self.phi_net(self.sample_next_states)
        if self.args.minus:
            self.expert_next_states = self.expert_next_states - self.expert_states
            self.sample_next_states = self.sample_next_states - self.sample_states
        
        update_step = self.args.f_epoch
        self.f_net.train()
        for _ in range(1, update_step):
            # if self.only_state:
            #     loss_f = (torch.mean(self.f_net(self.expert_states)) - 
            #             torch.mean(self.f_net(self.sample_states)))
            if self.state_action:
                loss_f = (torch.mean(torch.tanh(self.f_net(self.expert_states, self.expert_actions))) - torch.mean(torch.tanh(self.f_net(self.sample_states, self.sample_actions))))
                gradient_penalty_f = gradient_penalty(self.f_net,
                                                    torch.cat((self.expert_states, self.expert_actions), dim=-1),
                                                    torch.cat((self.sample_states, self.sample_actions), dim=-1), l=self.args.alpha)
            else:
                loss_f = (torch.mean(torch.tanh(self.f_net(self.expert_states, self.expert_next_states))) - torch.mean(torch.tanh(self.f_net(self.sample_states, self.sample_next_states))))
                gradient_penalty_f = gradient_penalty(self.f_net, 
                                                        torch.cat((self.expert_states, self.expert_next_states), dim=-1), 
                                                        torch.cat((self.sample_states, self.sample_next_states), dim=-1), l=self.args.alpha)
            
            total_loss_f = loss_f + gradient_penalty_f

            self.f_net.zero_grad()
            total_loss_f.backward()
            self.f_optimizer.step()
        
        # record f_loss and f_value
        self.f_net.eval()
        
        if self.state_action:
            loss_f = (torch.mean(self.f_net(self.expert_states, self.expert_actions)) 
                - torch.mean(self.f_net(self.sample_states, self.sample_actions)))
            f_value = torch.mean(self.f_net(self.expert_states, self.expert_actions)).item()
        else:
            loss_f = (torch.mean(self.f_net(self.expert_states, self.expert_next_states)) 
                    - torch.mean(self.f_net(self.sample_states, self.sample_next_states)))
            f_value = torch.mean(self.f_net(self.expert_states, self.expert_next_states)).item()
        wandb.log({'f_loss': loss_f.item(), 'f_value': f_value}, step=self.time_step)
        
        # empty the buffer of f_net update
        self.sample_states = []
        self.sample_next_states = []
        self.sample_actions = []
    
    def evaluation(self):
        avg_reward = round(self.sum_episodes_reward / self.sum_episodes_num, 2)
        wandb.log({'train_average_score': avg_reward}, step=self.time_step)
        self.sum_episodes_reward = 0
        self.sum_episodes_num = 0
    
    def generate_heat(self):
        if "maze2d-" not in self.args.env_name or self.state_action:
            print("self.args.env_name:", self.args.env_name)
            return
        else:
            print("plotting heat")
        # Define the input range and sampling density
        x_min, x_max = 0.5, 3.5
        y_min, y_max = 0.5, 3.5
        density = 0.02

        # Generate x and y coordinates
        x = np.arange(x_min, x_max, density)
        y = np.arange(y_min, y_max, density)
        X, Y = np.meshgrid(x, y)

        # Set f_net to evaluation mode
        self.f_net.eval()

        # Generate input data, first two are xy coordinates, last two are 00
        input_grid = np.c_[X.ravel(), Y.ravel(),  np.zeros_like(Y.ravel()), np.zeros_like(Y.ravel())]
        if self.args.using_icvf:
            input_tensor = self.phi_net(torch.tensor(input_grid, dtype=torch.float32).float().to(self.device))
        else:
            input_tensor = torch.tensor(input_grid, dtype=torch.float32).float().to(self.device)

        # Perform forward pass to compute the output
        with torch.no_grad():
            # if self.only_state:
            #     output_tensor = self.f_net(input_tensor)
            
            output_tensor = -self.f_net(input_tensor, input_tensor)
                
        # output_tensor = -(output_tensor- output_tensor.mean())/output_tensor.std()
        # output_tensor = torch.exp(output_tensor)

        # output_tensor = torch.sigmoid(output_tensor)
        # Assuming output is a scalar value
        Z = output_tensor.cpu().numpy().reshape(X.shape)

        # Generate heatmap
        plt.figure(figsize=(8, 6))
        plt.contourf(X, Y, Z, levels=100, cmap='viridis')
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=15)  # 增大colorbar的字体大小
        plt.title('Normal Space', fontsize=28)
        plt.xlabel('X Coordinate', fontsize=25)
        plt.ylabel('Y Coordinate', fontsize=25)
        
        plt.plot([1.5, 1.5], [0.5, 2.5], color='orange', linewidth=2.5)
        plt.plot([1.5, 2.5], [2.5, 2.5], color='orange', linewidth=2.5)
        plt.plot([2.5, 2.5], [0.5, 2.5], color='orange', linewidth=2.5)

        plt.xticks(fontsize=18)  # 增大x轴刻度字体大小
        plt.yticks(fontsize=18)  # 增大y轴刻度字体大小
        plt.tight_layout()


        # Get the current system time (hour and minute)
        current_time = datetime.now().strftime("%H%M")

        # Get the current timestep
        timestep = str(self.time_step)

        # Save the figure with the timestamp and time in the filename
        plt.savefig(f'visual/heat_rewd_{current_time}_{timestep}.png')
        print(f"Saved heatmap at timestep {self.time_step}")
        plt.close()

    # def generate_exp_heat(self):
    #     if "maze" not in self.args.env_name or self.state_action:
    #         return
    #     # Define the input range and sampling density
    #     x_min, x_max = 0, 5
    #     y_min, y_max = 0, 5
    #     density = 0.02

    #     x = np.arange(x_min, x_max, density)
    #     y = np.arange(y_min, y_max, density)
    #     X, Y = np.meshgrid(x, y)

    #     input_grid = np.c_[X.ravel(), Y.ravel()] 
    #     target = np.array([2, 3])
    #     distances = np.linalg.norm(input_grid - target, axis=1)
    #     output_values = np.exp(-distances)
    #     Z = output_values.reshape(X.shape)

    #     # Generate heatmap
    #     plt.figure(figsize=(8, 6))
    #     plt.contourf(X, Y, Z, levels=100, cmap='viridis')
    #     plt.colorbar(label='exp(-distance) to (2, 3)')
    #     plt.title('Heatmap of exp(-distance) to (2, 3)')
    #     plt.xlabel('x')
    #     plt.ylabel('y')

    #     # Save the figure with the timestamp and time in the filename
    #     plt.savefig(f'visual/exp_heat.png')
    #     plt.close()

    def evaluate_policy(self, eval_episodes=10):
        if 'dm_control' in self.args.env_name:
            env = gym.make(self.args.env_name)
            env = gym.wrappers.FlattenObservation(env)
        else:
            dataset = minari.load_dataset('mujoco/humanoid/expert-v0')
            env = dataset.recover_environment()
        avg_reward = 0.0
        avg_freward = 0.0
        avg_sig_freward = 0.0
        avg_episode_length = 0.0

        for ep in range(eval_episodes):
            # state = env.reset(seed = (self.args.seed + ep + self.time_step))
            (state,info), done = env.reset(), False
            episode_reward = 0.0
            episode_freward = 0.0
            episode_sig_freward = 0.0
            done = False
            episode_length = 0

            while not done:
                action = self.agent.select_action(state)

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = float(truncated or terminated)
                self.f_net.eval()
                state_tensor = torch.FloatTensor(state).to(self.device)
                next_state_tensor = torch.FloatTensor(next_state).to(self.device)
                if self.args.using_icvf:
                    state_tensor = self.phi_net(state_tensor)
                    next_state_tensor = self.phi_net(next_state_tensor)
                next_state_tensor = next_state_tensor - state_tensor
                fake_reward = -self.f_net(state_tensor, next_state_tensor)
                sig_fake_reward = torch.sigmoid(fake_reward)
                episode_freward += fake_reward.item()
                episode_sig_freward += sig_fake_reward.item()
                self.f_net.train()
                
                episode_reward += reward
                state = next_state
                episode_length += 1

            avg_reward += episode_reward
            avg_freward += episode_freward
            avg_sig_freward += episode_sig_freward
            avg_episode_length += episode_length

        avg_reward /= eval_episodes
        avg_freward /= eval_episodes
        avg_sig_freward /= eval_episodes
        avg_episode_length /= eval_episodes

        # normalized_score = get_normalized_score(self.args.env_name, avg_reward) * 100
        wandb.log({
            'eval_average_score': avg_reward / 10, 
            # 'eval_normalized_score': normalized_score,
            'eval_avg_episode_length': avg_episode_length,
            'eval_f_reward': avg_freward,
            'eval_sig_f_reward': avg_sig_freward
        }, step=self.time_step)
        print(f"Evaluation over {eval_episodes} episodes: Avg Reward = {avg_reward:.2f}, Avg Episode Length = {avg_episode_length:.2f}")

    def pretrain(self):
        self.get_random_dataset(self.args.update_timestep)
        print("Pretraining f_net")

        for epoch in range(2500):
            # only used for s and s'
            random_states = torch.stack(self.sample_states).float().to(self.device)
            random_next_states = torch.stack(self.sample_next_states).float().to(self.device)
            random_actions = torch.stack(self.sample_actions).float().to(self.device)
            
            expert_indices = torch.randint(0, len(self.expert_states_buffer), (len(self.sample_states),))
            self.expert_states = self.expert_states_buffer[expert_indices]
            self.expert_next_states = self.expert_next_states_buffer[expert_indices]
            self.expert_actions = self.expert_actions_buffer[expert_indices]
            if self.using_icvf:
                random_states = self.phi_net(random_states)
                random_next_states = self.phi_net(random_next_states)
                self.expert_states = self.phi_net(self.expert_states)
                self.expert_next_states = self.phi_net(self.expert_next_states)
            if self.state_action:
                random_batch = torch.cat((random_states, random_actions), dim=-1)
                expert_batch = torch.cat((self.expert_states, self.expert_actions), dim=-1)
            else:
                random_batch = torch.cat((random_states, random_next_states), dim=-1)
                expert_batch = torch.cat((self.expert_states, self.expert_next_states), dim=-1)
            
            
            loss_f = torch.mean(torch.tanh(self.f_net(expert_batch))) - torch.mean(torch.tanh(self.f_net(random_batch)))
            gradient_penalty_f = gradient_penalty(self.f_net, 
                                                  expert_batch,
                                                  random_batch, l=self.args.alpha)
            total_loss_f = loss_f + gradient_penalty_f

            # Optimize f_net by minimizing total_loss_f
            self.f_net.zero_grad()
            total_loss_f.backward()
            self.f_optimizer.step()
            wandb.log({'f_pretain_loss': loss_f.item()}, step=epoch)
        self.sample_states = []
        self.sample_next_states = []
        self.sample_actions = []
        self.expert_states = []
        self.expert_next_states = []
        self.expert_actions = []
        print("Pretraining done")
        self.generate_heat()

    def save_fnet(self):
        os.makedirs('model', exist_ok=True)
        env_firstname = self.args.env_name.split('-')[0]
        path = os.path.join('model', env_firstname + 'pref.pt')
        torch.save(self.f_net.state_dict(), path)
        print(f"Model saved to {path}")

    def get_random_dataset(self, num_samples):
        
        time_step = 0
        while True:
            # state = self.env.reset(seed=(time_step + self.args.seed))
            (state, info), done = self.env.reset(), False

            for _ in range(1, self.args.max_ep_len + 1):
                action = self.env.action_space.sample()
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = float(truncated or terminated)
                state = next_state
                time_step += 1
                
                self.sample_states.append(torch.FloatTensor(state))
                self.sample_next_states.append(torch.FloatTensor(next_state))
                self.sample_actions.append(torch.FloatTensor(action))
                if done:
                    break
            if time_step >= num_samples:
                break
        self.sample_states = self.sample_states[:self.args.update_timestep]
        self.sample_next_states = self.sample_next_states[:self.args.update_timestep]
        self.sample_actions = self.sample_actions[:self.args.update_timestep]
        
        
    def plot_reward(self):
        print("Plotting reward")
        lens = 1000
        obs = self.phi_net(self.expert_states_buffer[:lens])
        reward = self.dataset['rewards'][:lens]
        
        f_value = (-self.f_net(obs, obs)).detach().cpu().numpy()

        obs = obs.cpu().numpy()
        import matplotlib.colors as mcolors
        pca = PCA(n_components=2)
        reduced_observations = pca.fit_transform(obs)

        # First plot
        print("Plotting first plot")
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(reduced_observations[:, 0], reduced_observations[:, 1], c=reward, cmap='viridis', marker='o')
        plt.title('2D visualization of the original trajectory (PCA) with normalized rewards')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.colorbar(scatter, label='Normalized Reward')
        plt.savefig(f'visual/normal_rewards_{self.time_step}.png')  # Add time_step to the filename
        plt.close()

        # Second plot
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(reduced_observations[:, 0], reduced_observations[:, 1], c=f_value, cmap='viridis', marker='o')
        plt.title('2D visualization of the original trajectory (PCA) with f_value')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.colorbar(scatter, label='Normalized Reward')
        plt.savefig(f'visual/f_value_{self.time_step}.png')  # Add time_step to the filename
        plt.close()

        # Third plot
        state_replace = self.expert_states_buffer[:lens].cpu().numpy()
        reduced_observations = pca.fit_transform(state_replace)
        reward_replace = self.agent.buffer.reward[:lens].flatten()
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(reduced_observations[:, 0], reduced_observations[:, 1], c=reward_replace, cmap='viridis', marker='o')
        plt.title('2D visualization of the original trajectory (PCA) with fake reward')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.colorbar(scatter, label='Normalized Reward')
        plt.savefig(f'visual/fake_reward_{self.time_step}.png')  # Add time_step to the filename
        plt.close()

        print("Plotting done")
    
    def expand_and_fill(self, buffer, target_len):
        current_len = buffer.shape[0]
        if current_len < target_len:
            # compute repeat times
            repeat_count = target_len // current_len
            # repeat data
            repeated_buffer = buffer.repeat(repeat_count, 1)
            remaining_len = target_len - repeated_buffer.shape[0]
            if remaining_len > 0:
                # sample additional data
                additional_samples = buffer[torch.randint(0, current_len, (remaining_len,))]
                # concatenate additional samples
                expanded_buffer = torch.cat([repeated_buffer, additional_samples], dim=0)
            else:
                expanded_buffer = repeated_buffer
        else:
            expanded_buffer = buffer
        return expanded_buffer
