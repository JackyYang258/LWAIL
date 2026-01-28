
import os
import pickle


savedir = "one_expert_trajectory"
file_path = os.path.join(savedir, "hopper-expert-v2" + ".pkl")
with open(file_path, 'rb') as f:
    expert_dataset = pickle.load(f)
    
print(expert_dataset['observations'].shape)