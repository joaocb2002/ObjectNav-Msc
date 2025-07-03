import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F

# Deep Q-Network with embedded goal index and deeper architecture
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, goal_embedding_dim=8, num_goals=80):
        super(DQN, self).__init__()
        self.embedding = nn.Embedding(num_goals, goal_embedding_dim)

        # Adjust input dimension: remove goal_idx scalar and replace with embedding
        adjusted_input_dim = input_dim - 1 + goal_embedding_dim

        self.net = nn.Sequential(
            nn.Linear(adjusted_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),  # Bottleneck layer
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        state = x[:, :-1]           # all but last (goal ID)
        goal_idx = x[:, -1].long()  # ensure long type for embedding

        goal_emb = self.embedding(goal_idx)
        x = torch.cat([state, goal_emb], dim=1)
        return self.net(x)

# Replay Buffer for experience replay
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)

        states      = torch.from_numpy(np.stack(state)).float()
        actions     = torch.tensor(action, dtype=torch.long)
        rewards     = torch.tensor(reward, dtype=torch.float32)
        next_states = torch.from_numpy(np.stack(next_state)).float()
        dones       = torch.tensor(done, dtype=torch.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

# Epsilon-greedy action selection
def select_action(state, policy_net, epsilon, n_actions):
    if random.random() < epsilon:
        return random.randint(0, n_actions - 1)
    else:
        with torch.no_grad():
            state = torch.from_numpy(state).unsqueeze(0).float()
            q_values = policy_net(state)
            return q_values.argmax().item()
