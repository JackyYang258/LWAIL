import numpy as np

def generate_state_nextstate_dataset(target_position=(2, 3, 0, 0), position_tolerance=0.4, num_samples=2000):
    dataset = {
        'observations': [],
        'next_observations': []
    }
    
    for _ in range(num_samples):
        # Sample a state close to the target position
        state = np.array(target_position) + np.random.uniform(-position_tolerance, position_tolerance, size=4)
        
        # Apply some simple dynamics to generate the next state
        next_state = state + np.random.uniform(-0.1, 0.1, size=4)  # Small random change
        
        dataset['observations'].append(state)
        dataset['next_observations'].append(next_state)
    
    # Convert lists to numpy arrays
    for key in dataset.keys():
        dataset[key] = np.array(dataset[key])
    
    return dataset

if __name__ == "__main__":
    dataset = generate_state_nextstate_dataset()
    
    np.savez('dataset/state_nextstate_dataset.npz', **dataset)
    print("State-next state dataset generated and saved.")
