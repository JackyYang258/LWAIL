import torch
import os.path as osp

# Define the directory where the data is saved
SAVE_DIR = "/scratch/bdaw/kaiyan289/intentDICE/data/traj"
# Define the name of the file to load
save_name = "maze2d_open_dense_v0" + "_expert_dataset_%d_%d.pt" % (
    1000,
    100,
)

# Load the data
data_path = osp.join(SAVE_DIR, save_name)
data = torch.load(data_path)

# Print the loaded data
print("Loaded data from ", save_name)
print("Done:", data["done"])
print("Obs:", data["obs"])
print("Next Obs:", data["next_obs"])
print("Actions:", data["actions"])

# Additional info
print("Num episodes:", len(data["done"]))  # Assuming done indicates episode count
print("Num steps:", len(data["obs"]))