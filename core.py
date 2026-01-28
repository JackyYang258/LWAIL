import torch
from tqdm import tqdm

from network import FullyConnectedNet, PhiNet
from td3 import TD3
from ddpg import DDPG
from sac import SAC
from ppo import PPO
from utils import  gradient_penalty, get_normalized_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import matplotlib.pyplot as plt
import gym
from datetime import datetime
import wandb
import numpy as np
from d4rl_utils import FixedStartWrapper

class Agent:
    def __init__(self, state_dim, action_dim, env, expert_buffer, args):
        # Basic information
        self.args = args
        self.device = torch.device(self.args.cuda if torch.cuda.is_available() else 'cpu')
        self.using_icvf = args.using_icvf
        self.state_action = args.state_action
        self.expert_sample = True
        self.update_everystep = args.update_everystep
        max_action = float(env.action_space.high[0])
        self.agent_kind = args.downstream
        self.max_action = float(env.action_space.high[0])
        if self.agent_kind == 'td3':
            self.agent = TD3(state_dim, action_dim, self.args.lr_actor, self.args.lr_critic, self.device, max_action, self.args.curl)
        elif self.agent_kind == "ddpg":
            self.agent = DDPG(state_dim, action_dim, self.args.lr_actor, self.args.lr_critic, self.device)
        elif self.agent_kind == 'sac':
            self.agent = SAC(state_dim, action_dim, self.args.lr_actor, self.args.lr_critic, self.args.gamma, self.device)
        elif self.agent_kind == 'ppo': # added 2025/11/20
            self.agent = PPO(state_dim, action_dim, self.args.lr_actor, self.args.lr_critic, self.args.gamma, self.args, device=self.device)
        
        print("action_space:", env.action_space)
        print(f"max_action: {self.max_action}")
        self.env = env

        if 'antmaze' in args.env_name:
            self._init_normalization(expert_buffer)
            expert_obs_norm = self._normalize_obs(expert_buffer['observations'])
            expert_next_obs_norm = self._normalize_obs(expert_buffer['next_observations'])
        
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
            env_firstname = self.args.env_name.split('-')[0]
            if 'maze2d' in self.args.env_name:
                env_firstname = 'maze2d-medium'
            model_dir = os.path.join(os.getcwd(), "icvf_model")
            icvf_path = os.path.join(model_dir, f"{env_firstname}{args.dataset_num}{args.dataset_quality}.pt")
            self.phi_net.load_state_dict(torch.load(icvf_path, weights_only=False))
            for param in self.phi_net.parameters():
                param.requires_grad = False
            self.phi_net.to(self.device)
            print('Using ICVF')
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
        self.generate_exp_heat()
        self.pretrain()
        while self.time_step <= self.args.max_training_timesteps:
            state = self.env.reset()
            if 'antmaze' in self.args.env_name:
                state = self._normalize_obs(state)
            current_ep_reward = 0
            for _ in range(self.args.max_ep_len + 1):
                if self.agent_kind in ['td3', 'ddpg', 'sac', 'ppo']: # ppo added 2025/11/20
                    if self.agent_kind == 'ppo':
                        action, action_logprob, val = self.agent.select_action(state)
                        action = action.detach().cpu().numpy().squeeze()
                        # print("action:", action.shape)
                    else:
                        if self.time_step < self.args.start_timesteps:
                            action = self.env.action_space.sample()
                        else:
                            action = self.agent.select_action_withrandom(state)
                    # noise_scale = 0.5
                    # noise = np.random.normal(0, noise_scale, size=action.shape)
                    # noisy_action = action + noise
                    # noisy_action = np.clip(noisy_action, -self.max_action, self.max_action)
                    # next_state, reward, done, _ = self.env.step(noisy_action)
                    next_state, reward, done, _ = self.env.step(action)
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
                        
                    if self.agent_kind != "ppo": self.agent.buffer.add(state, action, next_state, fake_reward, float(done))
                    else: self.agent.buffer.store(state, action, action_logprob, val, next_state, fake_reward, float(done))
                    state = next_state
                elif self.agent_kind == 'onlytd3':
                    if self.time_step < self.args.start_timesteps:
                        action = self.env.action_space.sample()
                    else:
                        action = self.agent.select_action_withrandom(state)
                    next_state, reward, done, _ = self.env.step(action)
                    # reward = sigmoid(reward)
                    reward = torch.sigmoid(torch.tensor(reward)).item()
                    self.agent.buffer.add(state, action, next_state, reward, float(done))
                    state = next_state
                    
                        
                
                self.time_step += 1
                current_ep_reward += reward
                # print("time_step:", self.time_step)
                if self.time_step % self.args.update_timestep == 0:
                    
                    if self.agent_kind in ['td3', 'ddpg', 'sac']:
                        self.f_update()
                        self.get_pseudo_rewards()
                    if self.time_step % (self.args.update_timestep * 20) == 0:
                        self.generate_heat()    
                    
                    # self.get_pseudo_rewards() # Update the buffer with pseudo rewards
                    if self.agent_kind == 'ppo': # uncommented 2025/11/20
                         self.agent.update() # Update the agent with the pseudo rewards

                if self.time_step > self.args.update_timestep:
                    if self.agent_kind in ['td3', 'ddpg']:
                         self.agent.train()
                    elif self.agent_kind == 'sac' and self.time_step % 4096 == 0: 
                         self.agent.update()
                    
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
        for _ in range(update_step):
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
        normalized_score = get_normalized_score(self.args.env_name, avg_reward) * 100
        wandb.log({'train_average_score': avg_reward}, step=self.time_step)
        self.sum_episodes_reward = 0
        self.sum_episodes_num = 0
    
    def generate_heat(self):
        if "maze2d-medium" not in self.args.env_name or self.state_action:
            print("self.args.env_name:", self.args.env_name)
            return

        x_min, x_max = 0, 7
        y_min, y_max = 0, 7
        density = 0.02

        x = np.arange(x_min, x_max, density)
        y = np.arange(y_min, y_max, density)
        X, Y = np.meshgrid(x, y)

        self.f_net.eval()

        input_grid = np.c_[X.ravel(), Y.ravel(), np.zeros_like(Y.ravel()), np.zeros_like(Y.ravel())]

        if self.args.using_icvf:
            input_tensor = self.phi_net(torch.tensor(input_grid, dtype=torch.float32).to(self.device))
        else:
            input_tensor = torch.tensor(input_grid, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            output_tensor = -self.f_net(input_tensor, input_tensor)
            output_tensor = torch.sigmoid(output_tensor)

        Z = output_tensor.cpu().numpy().reshape(X.shape)

        plt.figure(figsize=(8, 6))
        plt.contourf(X, Y, Z, levels=100, cmap='viridis')
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=15)
        plt.title('medium_env_heatmap', fontsize=28)
        plt.xlabel('x', fontsize=25)
        plt.ylabel('y', fontsize=25)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()

        current_time = datetime.now().strftime("%H%M")
        timestep = str(self.time_step)
        plt.savefig(f'visual/heat_rewd_{current_time}_{timestep}.png')
        print(f"Saved heatmap at timestep {self.time_step}")
        plt.close()


    def generate_exp_heat(self):
        if "maze2d-medium" not in self.args.env_name or self.state_action:
            return

        x_min, x_max = 0, 7
        y_min, y_max = 0, 7
        density = 0.02

        x = np.arange(x_min, x_max, density)
        y = np.arange(y_min, y_max, density)
        X, Y = np.meshgrid(x, y)

        input_grid = np.c_[X.ravel(), Y.ravel()]
        target = np.array([6,6])  # 可以根据medium环境调整目标点
        distances = np.linalg.norm(input_grid - target, axis=1)
        output_values = np.exp(-distances)
        Z = output_values.reshape(X.shape)

        plt.figure(figsize=(8, 6))
        plt.contourf(X, Y, Z, levels=100, cmap='viridis')
        plt.colorbar(label='exp(-distance) to (2, 3)')
        plt.title('Heatmap of exp(-distance) to (2, 3)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(f'visual/exp_heat_medium.png')
        plt.close()

    def evaluate_policy(self, eval_episodes=10):
        env = gym.make(self.args.env_name)
        if 'maze2d' in self.args.env_name:
            env = FixedStartWrapper(env)
        avg_reward = 0.0
        avg_freward = 0.0
        avg_sig_freward = 0.0
        avg_episode_length = 0.0

        for ep in range(eval_episodes):
            # state = env.reset(seed = (self.args.seed + ep + self.time_step))
            state = env.reset()
            episode_reward = 0.0
            episode_freward = 0.0
            episode_sig_freward = 0.0
            done = False
            episode_length = 0

            while not done:
                if self.agent_kind != 'ppo': action = self.agent.select_action(state)
                else: 
                    action, _, __ = self.agent.select_action(state)
                    action = action.squeeze().detach().cpu().numpy()
                next_state, reward, done, _ = env.step(action)
                
                if 'antmaze' in self.args.env_name:
                    target_point = np.array([1, 8.5])
                    # Assumes the first two elements of the state are (x, y) coordinates
                    current_point = np.array([next_state[0], next_state[1]]) 
                    distance = np.linalg.norm(current_point - target_point)
                    
                    # Interpret "normalize to 0.1" as scaling an exp(-distance) reward
                    # This creates a reward in the range (0, 0.1]
                    reward = np.exp(-distance)

                self.f_net.eval()
                state_tensor = torch.FloatTensor(state).to(self.device)
                next_state_tensor = torch.FloatTensor(next_state).to(self.device)
                if self.args.using_icvf:
                    state_tensor = self.phi_net(state_tensor)
                    next_state_tensor = self.phi_net(next_state_tensor)
                next_state_tensor = next_state_tensor - state_tensor
                if self.state_action:
                    fake_reward = -self.f_net(state_tensor, torch.FloatTensor(action).to(self.device))
                else:
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

        normalized_score = get_normalized_score(self.args.env_name, avg_reward) * 100
        wandb.log({
            'eval_average_score': avg_reward, 
            'eval_normalized_score': normalized_score,
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
            state = self.env.reset()

            for _ in range(self.args.max_ep_len + 1):
                action = self.env.action_space.sample()
                next_state, reward, done, _ = self.env.step(action)
                state = next_state
                time_step += 1
                
                self.sample_states.append(torch.FloatTensor(state))
                self.sample_next_states.append(torch.FloatTensor(next_state))
                self.sample_actions.append(torch.FloatTensor(action))
                if done:
                    break
            if time_step >= num_samples:
                self.env.close()
                break
        self.sample_states = self.sample_states[:self.args.update_timestep]
        self.sample_next_states = self.sample_next_states[:self.args.update_timestep]
        self.sample_actions = self.sample_actions[:self.args.update_timestep]
        
        # while self.time_step <= 500000:
        #     state, _= self.env.reset()
        #     current_ep_reward = 0 # reward for the current episode

        #     for _ in range(1, self.args.max_ep_len + 1):
        #         if self.time_step < self.args.start_timesteps:
        #             action = self.env.action_space.sample()
        #         else:
        #             action = self.agent.select_action_withrandom(np.array(state))
        #         next_state, reward, terminated, truncated, _ = self.env.step(action)
        #         done = float(truncated or terminated)
        #         self.agent.buffer.add(state, action, next_state, reward, float(done))
        #         state = next_state

        #         self.time_step += 1
        #         current_ep_reward += reward

        #         if self.time_step % self.args.update_timestep == 0:
        #             self.get_pseudo_rewards() # Update the buffer with pseudo rewards

        #             if not self.update_everystep:
        #                 self.agent.update() # Update the agent with the pseudo rewards

        #         if self.update_everystep and self.time_step > self.args.update_timestep:
        #             self.agent.train()
                    
        #         if self.time_step % self.args.eval_freq == 0:
        #             self.evaluation()
        #             self.evaluate_policy()
                
        #         if done:
        #             break

        #     self.sum_episodes_reward += current_ep_reward
        #     self.sum_episodes_num += 1
        #     self.i_episode += 1
        
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
            repeat_count = target_len // current_len
            repeated_buffer = buffer.repeat(repeat_count, 1)
            remaining_len = target_len - repeated_buffer.shape[0]
            if remaining_len > 0:
                additional_samples = buffer[torch.randint(0, current_len, (remaining_len,))]
                expanded_buffer = torch.cat([repeated_buffer, additional_samples], dim=0)
            else:
                expanded_buffer = repeated_buffer
        else:
            expanded_buffer = buffer
        return expanded_buffer
    
    def _init_normalization(self, dataset):
        print("Initializing Normalization Statistics from Expert Buffer...")
        obs = dataset['observations']
        
        # 1. Compute Global Mean and Std
        self.obs_mean = np.mean(obs, axis=0)
        self.obs_std = np.std(obs, axis=0) + 1e-10
        
        # 2. Compute XY Min/Max for Coordinate Normalization
        # NOTE: Ideally, these should come from the FULL offline dataset, not just the expert buffer.
        # If you have access to the full D4RL dataset stats, replace these lines.
        self.xy_min = obs[:, :2].min(axis=0)
        self.xy_max = obs[:, :2].max(axis=0)
        
        print(f"Obs Mean: {self.obs_mean[:2]}...")
        print(f"XY Min: {self.xy_min}, XY Max: {self.xy_max}")

    def _normalize_obs(self, obs):
        """
        Applies the specific AntMaze processing:
        1. Standardize (mean/std)
        2. Normalize XY coordinates to [0, 1]
        """
        # Ensure obs is numpy (if it's a single state from gym, it might need reshape)
        is_single = False
        if len(obs.shape) == 1:
            obs = obs[np.newaxis, :]
            is_single = True
            
        # 1. Standardization
        # We only standardize dimensions present in mean/std (usually all)
        norm_obs = (obs - self.obs_mean) / self.obs_std
        
        # 2. XY Coordinate Normalization to [0, 1]
        # Only apply to the first 2 dimensions (x, y)
        norm_obs[:, :2] = (norm_obs[:, :2] - self.xy_min) / (self.xy_max - self.xy_min)
        
        if is_single:
            return norm_obs[0]
        return norm_obs
