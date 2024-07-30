import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity, state_size, action_size, device, seed, gamma=None, gae_lambda=None, num_envs=1):
        self.device = device
        self.state = torch.zeros(capacity, num_envs, state_size, dtype=torch.float).contiguous().pin_memory()
        self.action = torch.zeros(capacity, num_envs, action_size, dtype=torch.float).contiguous().pin_memory()
        self.reward = torch.zeros(capacity, num_envs, dtype=torch.float).contiguous().pin_memory()
        self.next_state = torch.zeros(capacity, num_envs, state_size, dtype=torch.float).contiguous().pin_memory()
        self.done = torch.zeros(capacity, num_envs, dtype=torch.int).contiguous().pin_memory()
        self.rng = np.random.default_rng(seed)
        self.idx = 0
        self.size = 0
        self.capacity = capacity
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.num_envs = num_envs

        # PPO-specific attributes
        if gamma is not None and gae_lambda is not None:
            self.advantage = torch.zeros(capacity, num_envs, dtype=torch.float).contiguous().pin_memory()
            self.value = torch.zeros(capacity, num_envs, dtype=torch.float).contiguous().pin_memory()
            self.log_prob = torch.zeros(capacity, num_envs, dtype=torch.float).contiguous().pin_memory()
            self.returns = torch.zeros(capacity, num_envs, dtype=torch.float).contiguous().pin_memory()
            self.to_device()

    def to_device(self):
        self.state = self.state.to(self.device)
        self.action = self.action.to(self.device)
        self.reward = self.reward.to(self.device)
        self.next_state = self.next_state.to(self.device)
        self.done = self.done.to(self.device)

        if self.gamma is not None and self.gae_lambda is not None:
            self.advantage = self.advantage.to(self.device)
            self.value = self.value.to(self.device)
            self.log_prob = self.log_prob.to(self.device)
            self.returns = self.returns.to(self.device)

    def __repr__(self) -> str:
        if self.gamma is not None and self.gae_lambda is not None:
            return 'PPOReplayBuffer'
        return 'NormalReplayBuffer'

    def add(self, transition):
        state, action, reward, next_state, done = transition[:5]
        self.state[self.idx] = torch.as_tensor(state)
        self.action[self.idx] = torch.as_tensor(action)
        self.reward[self.idx] = torch.as_tensor(reward)
        self.next_state[self.idx] = torch.as_tensor(next_state)
        self.done[self.idx] = torch.as_tensor(done)

        if self.gamma is not None and self.gae_lambda is not None:
            value, log_prob = transition[5:7]
            self.value[self.idx] = torch.as_tensor(value)
            self.log_prob[self.idx] = torch.as_tensor(log_prob)

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

    def sample(self, batch_size):
        sample_idxs = self.rng.choice(self.size, batch_size, replace=False)
        batch = (
            self.state[sample_idxs].to(self.device),
            self.action[sample_idxs].to(self.device),
            self.reward[sample_idxs].to(self.device),
            self.next_state[sample_idxs].to(self.device),
            self.done[sample_idxs].to(self.device)
        )
        return batch

    def clear(self):
        self.idx = 0
        self.size = 0

    def make_dataset(self):
        if self.gamma is not None and self.gae_lambda is not None:
            batch = (
                self.state[:self.size].flatten(0, 1),
                self.action[:self.size].flatten(0, 1),
                self.log_prob[:self.size].flatten(0, 1),
                self.value[:self.size].flatten(0, 1),
                self.advantage[:self.size].flatten(0, 1),
                self.returns[:self.size].flatten(0, 1)
            )
            return batch
        return None

    def get_next_values(self, agent, t) -> torch.Tensor:
        if self.gamma is None or self.gae_lambda is None:
            return None

        if self.num_envs > 1:
            if t != self.capacity - 1:
                next_values = torch.where(
                    self.done[t].bool(), 
                    agent.get_value(self.next_state[t], tensor=True).to(self.next_state.device),
                    self.value[t + 1]
                )
            else:
                next_values = agent.get_value(self.next_state[t], tensor=True).to(self.next_state.device)
        else:
            if t == self.capacity - 1 or self.done[t]:
                next_values = agent.get_value(self.next_state[t], tensor=True).to(self.next_state.device)
            else:
                next_values = self.value[t + 1]

        return next_values

    def compute_advantages_and_returns(self, agent) -> None:
        if self.gamma is None or self.gae_lambda is None:
            return None

        last_gae_lam = 0
        for t in reversed(range(self.capacity)):
            next_values = self.get_next_values(agent, t)
            delta = self.reward[t] + self.gamma * next_values - self.value[t]
            if t == self.capacity - 1:
                self.advantage[t] = delta
            else:
                self.advantage[t] = delta + self.gamma * self.gae_lambda * last_gae_lam

        self.returns = self.advantage + self.value