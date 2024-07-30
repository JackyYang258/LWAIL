import torch
from tqdm import tqdm

from network import network_weight_matrices
from ppo import PPO
from utils import time
import icecream as ic

using_ICVF = False

def train(expert_buffer, f_net, phi, env, seed, max_ep_len, max_training_timesteps, update_timestep, f_epoch, lr_f, action_std_decay_frequency, action_std_decay_rate, min_action_std, state_dim, action_dim, lr_actor, lr_critic, gamma, ppo_epochs, eps_clip, action_std_init=0.6):
    # def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std_init=0.6):
    time()
    agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, ppo_epochs, eps_clip, action_std_init)
    
    f_optimizer = torch.optim.Adam(f_net.parameters(), lr=lr_f)
    
    time_step = 0
    while time_step <= max_training_timesteps:
        
        state = env.reset()
        current_ep_reward = 0
        
        for step in tqdm(range(1, max_ep_len + 1)):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            agent.buffer.next_states.append(torch.tensor(next_state))
            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)
            
            time_step += 1
            current_ep_reward += reward
            
            # update if its time
            if time_step % update_timestep == 0:
                
                s1 = torch.tensor(expert_buffer['observations']).to(agent.device)
                s1_prime = torch.tensor(expert_buffer['next_observations']).to(agent.device)
                s2 = torch.squeeze(torch.stack(agent.buffer.states, dim=0)).detach().to(agent.device)
                s2_prime = torch.squeeze(torch.stack(agent.buffer.states, dim=0)).detach().to(agent.device)
                
                for f_step in range(1, f_epoch):
                    # Calculate the loss
                    
                    if using_ICVF:
                        loss_f = (torch.mean(f_net(phi(s2), phi(s2_prime))) - 
                                torch.mean(f_net(phi(s1), phi(s1_prime))))
                    else:
                        loss_f = (torch.mean(f_net(s2, s2_prime)) - torch.mean(f_net(s1, s1_prime)))
                    
                    # Optimize f_net by minimizing loss_f
                    f_net.zero_grad()
                    loss_f.backward()
                    f_optimizer.step()
                    
                    f_net = network_weight_matrices(f_net, 1)
                
                # agent.buffer.rewards = f_net(s,s')
                tensor_states = torch.stack(agent.buffer.states).to(agent.device).float()
                tensor_next_states = torch.stack(agent.buffer.next_states).to(agent.device).float()
                agent.buffer.rewards = f_net(tensor_states, tensor_next_states).tolist()
                print(agent.buffer.rewards)
                
                agent.update()
                agent.buffer.clear()
                
            if time_step % action_std_decay_frequency == 0:
                agent.decay_action_std(action_std_decay_rate, min_action_std)
                
            if done:
                break
    
    # evaluation

