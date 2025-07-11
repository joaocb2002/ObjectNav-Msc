import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F


class ObjectSearchDQN(nn.Module):
    def __init__(self, num_classes=27, num_actions=3, patch_size=9, goal_embedding_dim=16):
        super(ObjectSearchDQN, self).__init__()

        self.num_classes = num_classes
        self.num_actions = num_actions
        self.patch_size = patch_size
        self.goal_embedding_dim = goal_embedding_dim

        patch_flat_size = patch_size * patch_size
        belief_flat_size = patch_flat_size * num_classes

        # Goal embedding layer
        self.goal_embedding = nn.Embedding(num_classes, goal_embedding_dim)

        # Sub-networks
        self.pose_fc = nn.Linear(3, 64)
        self.occupancy_fc = nn.Linear(patch_flat_size, 128)
        self.belief_fc = nn.Linear(belief_flat_size, 512)

        combined_input_dim = 64 + 128 + 512 + goal_embedding_dim
        self.fc_combined = nn.Linear(combined_input_dim, 512)

        # Dueling DQN output layers
        self.fc_value = nn.Linear(512, 1)
        self.fc_advantage = nn.Linear(512, num_actions)

    def forward(self, pose_vec, occupancy_patch, belief_patch, goal_id):
        goal_emb = self.goal_embedding(goal_id)

        pose_feat = F.relu(self.pose_fc(pose_vec))
        occupancy_feat = F.relu(self.occupancy_fc(occupancy_patch))
        belief_feat = F.relu(self.belief_fc(belief_patch))

        combined = torch.cat([pose_feat, occupancy_feat, belief_feat, goal_emb], dim=-1)
        combined_feat = F.relu(self.fc_combined(combined))

        value = self.fc_value(combined_feat)
        advantage = self.fc_advantage(combined_feat)

        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, pose, occupancy, belief, goal_id, action, reward, next_pose, next_occupancy, next_belief, next_goal_id, done):
        self.buffer.append((pose, occupancy, belief, goal_id, action, reward, next_pose, next_occupancy, next_belief, next_goal_id, done))

    def sample(self, batch_size, device="cuda:0"):
        batch = random.sample(self.buffer, batch_size)
        pose, occupancy, belief, goal_id, action, reward, next_pose, next_occupancy, next_belief, next_goal_id, done = zip(*batch)

        return (
            torch.from_numpy(np.stack(pose)).float().to(device),
            torch.from_numpy(np.stack(occupancy)).float().to(device),
            torch.from_numpy(np.stack(belief)).float().to(device),
            torch.tensor(goal_id, dtype=torch.long).to(device),
            torch.tensor(action, dtype=torch.long).to(device),
            torch.tensor(reward, dtype=torch.float32).to(device),
            torch.from_numpy(np.stack(next_pose)).float().to(device),
            torch.from_numpy(np.stack(next_occupancy)).float().to(device),
            torch.from_numpy(np.stack(next_belief)).float().to(device),
            torch.tensor(next_goal_id, dtype=torch.long).to(device),
            torch.tensor(done, dtype=torch.float32).to(device),
        )


    def __len__(self):
        return len(self.buffer)

def select_action(pose_vec, occupancy_patch, belief_patch, goal_id, policy_net, epsilon, n_actions):
    if random.random() < epsilon:
        return random.randint(0, n_actions - 1)
    else:
        with torch.no_grad():
            q_values = policy_net(pose_vec.unsqueeze(0),
                                  occupancy_patch.unsqueeze(0),
                                  belief_patch.unsqueeze(0),
                                  goal_id.unsqueeze(0))
            return q_values.argmax().item()
