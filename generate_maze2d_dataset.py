import torch
import numpy as np
import gym

# Set device to CPU or CUDA
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6):
        self.has_continuous_action_space = has_continuous_action_space
        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob, state_val = self.policy_old.act(state)
        return action.detach().cpu().numpy().flatten() if self.has_continuous_action_space else action.item()

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

def generate_expert_dataset(env_name, model_path, max_episodes=1000, max_steps_per_episode=150, max_transitions=990000):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] if isinstance(env.action_space, gym.spaces.Box) else env.action_space.n
    has_continuous_action_space = isinstance(env.action_space, gym.spaces.Box)

    ppo = PPO(state_dim, action_dim, lr_actor=0.0003, lr_critic=0.001, gamma=0.99, K_epochs=80, eps_clip=0.2, 
              has_continuous_action_space=has_continuous_action_space)
    ppo.load(model_path)

    dataset = {
        'observations': [],
        'actions': [],
        'next_observations': [],
        'rewards': [],
        'terminals': [],
        'dones_float': []
    }

    total_transitions = 0

    for episode in range(max_episodes):
        if total_transitions >= max_transitions:
            break

        state = env.reset()
        done = False
        step_count = 0

        while not done and step_count < max_steps_per_episode:
            action = ppo.select_action(state)
            next_state, reward, done, _ = env.step(action)

            dataset['observations'].append(state)
            dataset['actions'].append(action)
            dataset['next_observations'].append(next_state)
            dataset['rewards'].append(reward)
            dataset['terminals'].append(float(done))
            dataset['dones_float'].append(float(done))

            state = next_state
            step_count += 1
            total_transitions += 1

            if total_transitions >= max_transitions:
                break

        if step_count >= max_steps_per_episode:
            done = True
            dataset['terminals'][-1] = float(done)
            dataset['dones_float'][-1] = float(done)
        
        if total_transitions >= max_transitions:
            break
    
    # Convert lists to numpy arrays
    for key in dataset.keys():
        dataset[key] = np.array(dataset[key])

    return dataset

if __name__ == "__main__":
    env_name = 'Pendulum-v1'  # Replace with your environment
    model_path = 'path_to_pretrained_model.pth'
    dataset = generate_expert_dataset(env_name, model_path)

    # Save dataset to a file
    np.savez('expert_dataset.npz', **dataset)
    print("Expert dataset generated and saved.")
