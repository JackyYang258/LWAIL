import gym
import pickle
import numpy as np
import d4rl

class RandomDatasetGenerator:
    def __init__(self, env_name, num_samples, max_ep_len, seed):
        self.env = gym.make(env_name)
        self.num_samples = num_samples
        self.max_ep_len = max_ep_len
        self.seed = seed
        self.dataset = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'next_observations': [],
            'dones': []
        }

    def generate(self):
        time_step = 0
        done_positions = []

        while time_step < self.num_samples:
            state = self.env.reset(seed=(time_step + self.seed))

            for step in range(1, self.max_ep_len + 1):
                action = self.env.action_space.sample()
                next_state, reward, done, _ = self.env.step(action)

                # Append the data to the dictionary
                self.dataset['observations'].append(state)
                self.dataset['actions'].append(action)
                self.dataset['rewards'].append(reward)
                self.dataset['next_observations'].append(next_state)
                self.dataset['dones'].append(done)
                
                state = next_state
                time_step += 1

                if done:
                    # Print the time step where 'done' is True
                    print(f"'done' at time step: {time_step}")
                    done_positions.append(time_step)
                    break

                if time_step >= self.num_samples:
                    break

        # Convert lists to numpy arrays for consistency, like in d4rl
        self.dataset['observations'] = np.array(self.dataset['observations'])
        self.dataset['actions'] = np.array(self.dataset['actions'])
        self.dataset['rewards'] = np.array(self.dataset['rewards'])
        self.dataset['next_observations'] = np.array(self.dataset['next_observations'])
        self.dataset['dones'] = np.array(self.dataset['dones'])

        self.env.close()
        return self.dataset, done_positions

    def save(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self.dataset, f)


if __name__ == "__main__":
    env_name = "halfcheetah-expert-v2"  # Replace with any desired environment, e.g., 'd4rl:maze2d-random-v0'
    num_samples = 1000000
    max_ep_len = 1000
    seed = 42
    
    generator = RandomDatasetGenerator(env_name, num_samples, max_ep_len, seed)
    dataset, done_positions = generator.generate()
    
    # Save dataset to a pickle file
    generator.save("random_dataset.pkl")
    
    # Print out the positions where 'done' occurred
    print(f"Generated dataset saved to 'random_dataset.pkl'.")
    print(f"Done positions: {done_positions}")
