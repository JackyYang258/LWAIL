import matplotlib.pyplot as plt
import pickle

file_path = "/home/kaiyan3/siqi/IntentDICE/multiple_expert_trajectory/maze2d-umaze-dense-v1.pkl"
with open(file_path, 'rb') as f:
    dataset = pickle.load(f)
x = dataset['observations'][:5000][:,0]
y = dataset['observations'][:5000][:,1]
plt.scatter(x, y, c='blue', marker='o')

plt.title("Expert States Plot (from PyTorch Tensor)")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.savefig(f'visual/points')  # Add time_step to the filename
plt.close()