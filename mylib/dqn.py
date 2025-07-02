import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, goal_embedding_dim=8, num_goals=80):
        super(DQN, self).__init__()
        self.embedding = nn.Embedding(num_goals, goal_embedding_dim)

        self.fc1 = nn.Linear(input_dim - 1 + goal_embedding_dim, 128)  # exclude goal_idx scalar
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, output_dim)

    def forward(self, x):
        state = x[:, :-1]                 # all but last (goal ID)
        goal_idx = x[:, -1].long()        # goal ID as long

        goal_emb = self.embedding(goal_idx)
        x = torch.cat([state, goal_emb], dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

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
    
    
def select_action(state, policy_net, epsilon, n_actions):
    if random.random() < epsilon:
        return random.randint(0, n_actions - 1)
    else:
        with torch.no_grad():
            state = torch.from_numpy(state).unsqueeze(0).float()
            q_values = policy_net(state)
            return q_values.argmax().item()

