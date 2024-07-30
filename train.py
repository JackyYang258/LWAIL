import torch
from network import network_weight_matrices
from ppo import PPO

def train(expert_buffer, f_net, phi, env, seed, max_ep_len, max_training_timesteps, update_timestep, f_epoch, lr_f, action_std_decay_frequency, action_std_decay_rate, min_action_std, state_dim, action_dim, lr_actor, lr_critic, gamma, ppo_epochs, eps_clip, action_std_init=0.6):
    # def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std_init=0.6):
    agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, ppo_epochs, eps_clip, action_std_init)
    
    f_optimizer = torch.optim.Adam(f_net.parameters(), lr=lr_f)
    
    time_step = 0
    while time_step <= max_training_timesteps:
        
        state, _ = env.reset(seed=seed)
        current_ep_reward = 0
        
        for step in range(1, max_ep_len + 1):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            agent.buffer.next_states.append(next_state)
            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)
            
            time_step += 1
            current_ep_reward += reward
            
            # update if its time
            if time_step % update_timestep == 0:
                s1, s1_prime = expert_buffer['states'], expert_buffer['next_states']
                s2, s2_prime = agent.buffer['states'], agent.buffer['next_states']
                for f_step in range(1, f_epoch):
                    # Calculate the loss
                    loss_f = (torch.mean(f_net(phi(s2), phi(s2_prime))) - 
                            torch.mean(f_net(phi(s1), phi(s1_prime))))
                    
                    # Optimize f_net by minimizing loss_f
                    f_net.zero_grad()
                    loss_f.backward()
                    f_optimizer.step()
                    
                    f_net = network_weight_matrices(f_net, 1)
                
                # agent.buffer.rewards = f_net(s,s')
                agent.buffer.rewards = f_net(agent.buffer.states, agent.buffer.next_states)
                
                agent.update()
                agent.buffer.clear()
                
            if time_step % action_std_decay_frequency == 0:
                agent.decay_action_std(action_std_decay_rate, min_action_std)
                
            if done:
                break
    
    # evaluation

