import torch
import torch.nn as nn

class FullyConnectedNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims, activation=nn.ReLU, activate_final=False):
        super(FullyConnectedNet, self).__init__()
        layers = []
        current_dim = input_dim  # Since we will concatenate two inputs
        for dim in hidden_dims:
            layers.append(nn.Linear(current_dim, dim))
            layers.append(activation())
            current_dim = dim
        layers.append(nn.Linear(current_dim, 1))  # Output dimension is 1
        if activate_final:
            layers.append(activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x1, x2 = None):
        # Concatenate the inputs along the feature dimension
        if x2 is None:
            x = x1
        else:
            x = torch.cat((x1, x2), dim=-1)
        return self.net(x)
    
class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, state_action):
        reward = self.model(state_action)
        return reward

# class FullyConnectedNet(nn.Module):
#     def __init__(self, input_dim: int, hidden_dims, activation=nn.ReLU, activate_final=False):
#         super(FullyConnectedNet, self).__init__()
#         layers = []
#         current_dim = input_dim  
#         layers.append(nn.Linear(current_dim, 1))
#         self.net = nn.Sequential(*layers)

#     def forward(self, x1, x2):
#         # Concatenate the inputs along the feature dimension
#         x = torch.cat((x1, x2), dim=-1)
#         return self.net(x)

class PhiNet(nn.Module):
    def __init__(self, hidden_dims, activation=nn.GELU, activate_final=False):
        super(PhiNet, self).__init__()
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            if i + 1 < len(hidden_dims) or activate_final:
                layers.append(activation())
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x) 
    
def network_weight_matrices(model, max_norm, eps=1e-8):
    for module in model.modules():
        if isinstance(module, nn.Linear):
            w = module.weight.data
            spectral_norm = torch.linalg.matrix_norm(w, ord=2)
            denom = max(1, spectral_norm / max_norm)
            module.weight.data = w / (denom + eps)
    return model

class ContrastiveEncoder(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        middle_size = 128
        self.net = nn.Sequential(
            nn.Linear(input_size, middle_size),
            nn.ReLU(),
            nn.Linear(middle_size, middle_size),
            nn.ReLU(),
            nn.Linear(middle_size, output_size))
    
    def forward(self, s):
        return self.net(s)

class Contrastive_PD(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.feature_size = 32
        self.encoder = ContrastiveEncoder(input_size, self.feature_size)
        self.W = torch.nn.Parameter(torch.rand(self.feature_size, self.feature_size)) # note the "distance" is euclidean in embedding space; W does not have to be semi positive-definite
        
    def encode(self, x):
        v = self.encoder(x)
        return v / torch.norm(v, dim=-1, keepdim=True)
        
    def forward(self, s1, s2):
        z1, z2 = self.encode(s1), self.encode(s2)
        # logits = torch.matmul(z1, torch.matmul(self.W, z2.T))
        #print("logits-before:", logits)
        W2 = torch.matmul(F.softplus(self.W), F.softplus(self.W.T))
        logits = torch.matmul(z1, torch.matmul(W2, z2.T))
        # print("logit shape:", logits.shape)
        #print("logits-before:", logits)
        logits -= torch.max(logits, 1)[0][:, None]
        #print("logits-after:", logits)
        return logits 

# def _get_batch(iterator, loader):
#     try:
#         samples = next(iterator)
#     except StopIteration:
#         iterator = iter(loader)
#         samples = next(iterator)

#     samples = samples[0][:,None]
#     return samples, iterator


# def _eval_wasserstein_model(p_loader, q_loader, h, device):
#     p_length = len(p_loader.dataset)
#     q_length = len(q_loader.dataset)

#     p_length_inv = torch.DoubleTensor([1.]).to(device) / p_length
#     q_length_inv = torch.DoubleTensor([1.]).to(device) / q_length
#     h_p_expectation = torch.DoubleTensor([0.])
#     h_q_expectation = torch.DoubleTensor([0.])
#     with torch.no_grad():
#         for data in p_loader:
#             data = data[0][:,None].to(device)
#             h_p_expectation += torch.sum(h(data).double() * p_length_inv).item()

#         for data in q_loader:
#             data = data[0][:,None].to(device)
#             h_q_expectation += torch.sum(h(data).double() * q_length_inv).item()

#     return (h_p_expectation - h_q_expectation).item()


# def estimate_wasserstein_kantorovich_rubinstein(samples_p, samples_q, device, eval_steps=1000,
#                                                 bs=1024, eval_bs=2**15, optim='adam', lr=0.001,
#                                                 schedule='none', steps=100_00):
#     h = FullyConnectedNet(1, 1)
#     h.to(device)

#     h = network_weight_matrices(h, 1)

#     if optim == 'sgd':
#         optimizer = torch.optim.SGD(h.parameters(), lr=lr, momentum=0.9)
#     elif optim == 'adam':
#         optimizer = torch.optim.Adam(h.parameters(), lr=lr)
#     else:
#         raise NotImplementedError()

#     if schedule == 'cosine':
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
#     elif schedule == 'none' or schedule is None:
#         scheduler = None
#     else:
#         raise NotImplementedError()

#     #prepare the dataset
#     obs = phi(observations)
#     next_observations = phi(next_observations)

#     for i in trange(steps):

#         p_samples, q_samples = p_samples.to(device), q_samples.to(device)

#         optimizer.zero_grad()
#         output_p = h(p_samples)
#         output_q = h(q_samples)

#         #minus loss as we have to maximize
#         loss = -(torch.mean(output_p) - torch.mean(output_q))
#         loss.backward()
#         optimizer.step()

#         if scheduler is not None:
#             scheduler.step()

#         h = network_weight_matrices(h, 1)

#         if (i % eval_steps) == 0:
#             current_d_estimate = _eval_wasserstein_model(eval_p_loader, eval_q_loader, h, device)
#             print(f'Step {i}: Wasserstein-1 estimate {current_d_estimate:.5f}')

#     current_d_estimate = _eval_wasserstein_model(eval_p_loader, eval_q_loader, h, device)
#     print(f'Final Wasserstein-1 estimate {current_d_estimate:.5f}')
#     return current_d_estimate