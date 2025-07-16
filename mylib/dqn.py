import torch
import torch.nn as nn
from collections import deque
import random

class ObjectSearchAgent(nn.Module):
    def __init__(self, num_classes=27, num_actions=3, patch_size=9, goal_embedding_dim=32):
        super(ObjectSearchAgent, self).__init__()

        self.num_classes = num_classes
        self.num_actions = num_actions
        self.patch_size = patch_size
        self.goal_embedding_dim = goal_embedding_dim

        # Goal embedding
        self.goal_embedding = nn.Embedding(num_classes, goal_embedding_dim)

        # Pose encoder (x_norm, y_norm, orientation_norm)
        self.pose_fc = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

        # Occupancy encoder: CNN
        self.occupancy_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Belief map encoder: CNN
        self.belief_cnn = nn.Sequential(
            nn.Conv2d(num_classes, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Calculate CNN output size dynamically
        sample_tensor = torch.zeros(1, 1, patch_size, patch_size)
        occ_out_dim = self.occupancy_cnn(sample_tensor).shape[1]
        belief_sample_tensor = torch.zeros(1, num_classes, patch_size, patch_size)
        belief_out_dim = self.belief_cnn(belief_sample_tensor).shape[1]

        # Combined feature size
        combined_size = 128 + occ_out_dim + belief_out_dim + goal_embedding_dim

        # LSTM
        self.lstm = nn.LSTM(combined_size, 512, batch_first=True, num_layers=2)

        # Dueling DQN heads
        self.fc_value = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.fc_advantage = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, pose, occupancy_patch, belief_patch, goal, hidden_state=None):
        # Encode inputs
        pose_encoded = self.pose_fc(pose)
        occ_encoded = self.occupancy_cnn(occupancy_patch.unsqueeze(1))
        belief_encoded = self.belief_cnn(belief_patch)
        goal_encoded = self.goal_embedding(goal)

        # Concatenate features
        fused = torch.cat([pose_encoded, occ_encoded, belief_encoded, goal_encoded], dim=1)
        fused = fused.unsqueeze(1)  # For LSTM (batch_size, seq_len=1, features)

        lstm_out, hidden_state = self.lstm(fused, hidden_state)

        value = self.fc_value(lstm_out[:, -1])
        advantage = self.fc_advantage(lstm_out[:, -1])
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q_values, hidden_state

class SequenceReplayBuffer:
    def __init__(self, capacity=1000, sequence_length=10):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.buffer = []

    def push_episode(self, episode):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(episode)

    def sample(self, batch_size, device="cuda:0"):
        episodes = random.sample(self.buffer, batch_size)
        batch = []
        for ep in episodes:
            if len(ep) >= self.sequence_length:
                start_idx = random.randint(0, len(ep) - self.sequence_length)
                batch.append(ep[start_idx:start_idx + self.sequence_length])
            else:
                batch.append(ep)

        # Transpose and pad sequences
        transposed = list(zip(*[[(torch.tensor(item[0]),
                                  torch.tensor(item[1]),
                                  torch.tensor(item[2]),
                                  torch.tensor(item[3]),
                                  torch.tensor(item[4]),
                                  torch.tensor(item[5]),
                                  torch.tensor(item[6]),
                                  torch.tensor(item[7]),
                                  torch.tensor(item[8]),
                                  torch.tensor(item[9]),
                                  torch.tensor(item[10])) for item in seq] for seq in batch]))

        batched_tensors = []
        for group in transposed:
            batched_tensors.append(torch.stack([torch.stack(step) for step in group]).to(device))

        return batched_tensors

    def __len__(self):
        return len(self.buffer)
    
def select_action(policy_net, pose, occupancy_patch, belief_patch, goal, num_actions, hidden_state=None, epsilon=0.1):
    """
    Select an action using epsilon-greedy policy.
    Args:
        policy_net: The DQN network.
        pose: The current pose of the agent (tensor).
        occupancy_patch: The occupancy patch (tensor).
        belief_patch: The belief patch (tensor).
        goal: The target object ID (tensor).
        num_actions: Number of possible actions.
        hidden_state: Hidden state for LSTM (optional).
        epsilon: Epsilon value for exploration.
    Returns:
        action: Selected action index.
        hidden_state: Updated hidden state.
    """
    if random.random() < epsilon:
        action = random.randint(0, num_actions - 1)
        return action, hidden_state
    else:
        with torch.no_grad():
            q_values, hidden_state = policy_net(pose, occupancy_patch, belief_patch, goal, hidden_state)
            action = q_values.argmax().item()
        return action, hidden_state
