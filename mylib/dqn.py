import torch
import torch.nn as nn
from collections import deque
import random

class ObjectSearchAgent(nn.Module):
    def __init__(self, num_classes=27, num_actions=3, patch_size=9, goal_embedding_dim=32):
        super(ObjectSearchAgent, self).__init__()

        # Init parameters
        self.num_classes = num_classes
        self.num_actions = num_actions
        self.patch_size = patch_size
        self.goal_embedding_dim = goal_embedding_dim

        self.goal_embedding = nn.Embedding(num_classes, goal_embedding_dim)

        self.pose_fc = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

        self.occupancy_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.belief_cnn = nn.Sequential(
            nn.Conv2d(num_classes, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Sample tensors to determine output dimensions
        sample_tensor = torch.zeros(1, 1, patch_size, patch_size)
        occ_out_dim = self.occupancy_cnn(sample_tensor).shape[1]
        belief_sample_tensor = torch.zeros(1, num_classes, patch_size, patch_size)
        belief_out_dim = self.belief_cnn(belief_sample_tensor).shape[1]

        # Calculate combined size for LSTM input
        combined_size = 128 + occ_out_dim + belief_out_dim + goal_embedding_dim

        self.lstm = nn.LSTM(combined_size, 512, batch_first=True, num_layers=2)

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

        pose, occupancy_patch, belief_patch, goal = self._normalize_inputs(pose, occupancy_patch, belief_patch, goal)

        # Get batch and sequence dimensions
        batch_size, seq_len = pose.shape[:2]

        # Reshape for feature encoders
        pose_flat = pose.reshape(batch_size * seq_len, -1)
        occ_flat = occupancy_patch.reshape(batch_size * seq_len, *occupancy_patch.shape[2:])
        belief_flat = belief_patch.reshape(batch_size * seq_len, *belief_patch.shape[2:])
        goal_flat = goal.reshape(batch_size * seq_len)

        # Encode features
        pose_encoded = self.pose_fc(pose_flat)
        occ_encoded = self.occupancy_cnn(occ_flat)
        belief_encoded = self.belief_cnn(belief_flat)
        goal_encoded = self.goal_embedding(goal_flat)

        # Reshape encoded features to match LSTM input
        fused = torch.cat([pose_encoded, occ_encoded, belief_encoded, goal_encoded], dim=1)
        fused = fused.view(batch_size, seq_len, -1)

        lstm_out, hidden_state = self.lstm(fused, hidden_state)

        last_output = lstm_out  # Use all timesteps

        value = self.fc_value(last_output)
        advantage = self.fc_advantage(last_output)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q_values, hidden_state

    def _normalize_inputs(self, pose, occupancy_patch, belief_patch, goal):
        """ Check if batch and sequence dimensions are missing and add them """

        # One step input
        if pose.dim() == 1:
            pose = pose.unsqueeze(0).unsqueeze(0)
            occupancy_patch = occupancy_patch.unsqueeze(0).unsqueeze(0)
            belief_patch = belief_patch.unsqueeze(0).unsqueeze(0)
            goal = goal.unsqueeze(0).unsqueeze(0)

        # Batch input
        elif pose.dim() == 2:
            pose = pose.unsqueeze(0)
            occupancy_patch = occupancy_patch.unsqueeze(0)
            belief_patch = belief_patch.unsqueeze(0)
            goal = goal.unsqueeze(0)

        return pose, occupancy_patch, belief_patch, goal

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
                # Pad shorter sequences if needed (optional)
                batch.append(ep + [ep[-1]] * (self.sequence_length - len(ep)))

        # Now, batch is a list of length batch_size, each element is a list of sequence_length tuples

        # Unzip all elements
        poses, occ_patches, belief_patches, target_object_ids, actions, rewards, next_poses, next_occ_patches, next_belief_patches, next_target_object_ids, dones = zip(
            *[step for episode in batch for step in episode]
        )

        # Reshape to [batch_size, sequence_length, ...]
        def stack_and_reshape(tensors, shape):
            stacked = torch.stack(tensors).to(device)
            return stacked.view(batch_size, self.sequence_length, *shape)

        poses = stack_and_reshape(poses, poses[0].shape)
        occ_patches = stack_and_reshape(occ_patches, occ_patches[0].shape)
        belief_patches = stack_and_reshape(belief_patches, belief_patches[0].shape)
        target_object_ids = stack_and_reshape(target_object_ids, target_object_ids[0].shape)
        actions = torch.tensor(actions, device=device).view(batch_size, self.sequence_length)
        rewards = torch.tensor(rewards, device=device).view(batch_size, self.sequence_length)
        next_poses = stack_and_reshape(next_poses, next_poses[0].shape)
        next_occ_patches = stack_and_reshape(next_occ_patches, next_occ_patches[0].shape)
        next_belief_patches = stack_and_reshape(next_belief_patches, next_belief_patches[0].shape)
        next_target_object_ids = stack_and_reshape(next_target_object_ids, next_target_object_ids[0].shape)
        dones = torch.tensor(dones, dtype=torch.bool, device=device).view(batch_size, self.sequence_length)

        return (poses, occ_patches, belief_patches, target_object_ids,
                actions, rewards, next_poses, next_occ_patches,
                next_belief_patches, next_target_object_ids, dones)


    def __len__(self):
        return len(self.buffer)
    
def select_action(policy_net, pose, occupancy_patch, belief_patch, goal, num_actions, hidden_state, epsilon=0.1):
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
