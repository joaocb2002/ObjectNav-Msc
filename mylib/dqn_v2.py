import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random

class ObjectSearchAgent(nn.Module):
    def __init__(self, num_classes=28, num_actions=3, patch_size=11):
        super(ObjectSearchAgent, self).__init__()

        self.num_classes = num_classes
        self.num_actions = num_actions
        self.patch_size = patch_size

        # CNN for occupancy map (1 channel)
        self.cnn_occ = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # → 5x5
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # CNN for belief map (28 channels)
        self.cnn_belief = nn.Sequential(
            nn.Conv2d(num_classes, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # → 5x5
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        cnn_output_dim = (64 + 128) * 5 * 5  # = 192 * 5 * 5 = 4800

        # MLP for flat input (orientation + position + goal ID)
        self.mlp_flat = nn.Sequential(
            nn.Linear(4 + 2 + num_classes, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # Shared hidden layer
        self.shared_fc = nn.Sequential(
            nn.Linear(cnn_output_dim + 128, 512),
            nn.ReLU()
        )

        # Dueling streams
        self.value_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_actions)
        )


    def forward(self, pose, occupancy_patch, belief_patch, goal_id):
        """
        pose:       tensor of shape (batch, 6) → [4 orientation + 2 pos]
        occupancy_patch: tensor of shape (batch, 1, 11, 11)
        belief_patch:    tensor of shape (batch, 28, 11, 11)
        goal_id:    tensor of shape (batch, 28)
        """

        # Normalize inputs
        pose, occupancy_patch, belief_patch, goal_id = self._normalize_inputs(pose, occupancy_patch, belief_patch, goal_id)

        # Merge spatial inputs
        occ_feat = self.cnn_occ(occupancy_patch)        # → (batch, 64, 5, 5)
        belief_feat = self.cnn_belief(belief_patch)     # → (batch, 128, 5, 5)
        x_spatial = torch.cat([occ_feat, belief_feat], dim=1)  # → (batch, 192, 5, 5)
        x_spatial = torch.flatten(x_spatial, start_dim=1)  # → (batch, N)

        # Flat features
        x_flat = torch.cat([pose, goal_id], dim=1)  # → (batch, 6 + 28)
        x_flat = self.mlp_flat(x_flat)

        x = torch.cat([x_spatial, x_flat], dim=1)
        x = self.shared_fc(x)

        value = self.value_stream(x)               # (batch, 1)
        advantage = self.advantage_stream(x)       # (batch, num_actions)

        # Combine value and advantage into Q-values
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


    def _normalize_inputs(self, pose, occupancy_patch, belief_patch, goal):
        """ Check if batch and sequence dimensions are missing and add them """

        # Remove extra batch dimension if present
        if pose.dim() == 3:
            pose = pose.squeeze(0)  # remove batch dimension
            occupancy_patch = occupancy_patch.squeeze(0)
            belief_patch = belief_patch.squeeze(0)
            goal = goal.squeeze(0)

        # Add batch dim
        elif pose.dim() == 1:
            pose = pose.unsqueeze(0)
            occupancy_patch = occupancy_patch.unsqueeze(0)
            belief_patch = belief_patch.unsqueeze(0)
            goal = goal.unsqueeze(0)

        return pose, occupancy_patch, belief_patch, goal
    

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Unpack and stack each element of the state tuple
        poses, occs, beliefs, goals = zip(*states)
        next_poses, next_occs, next_beliefs, next_goals = zip(*next_states)

        return (
            (
                torch.stack(poses),
                torch.stack(occs),
                torch.stack(beliefs),
                torch.stack(goals)
            ),
            torch.tensor(actions, dtype=torch.long, device=poses[0].device),
            torch.tensor(rewards, dtype=torch.float32, device=poses[0].device),
            (
                torch.stack(next_poses),
                torch.stack(next_occs),
                torch.stack(next_beliefs),
                torch.stack(next_goals)
            ),
            torch.tensor(dones, dtype=torch.float32, device=poses[0].device)
        )

    def __len__(self):
        return len(self.buffer)

def select_action(policy_net, state, epsilon, num_actions):
    if random.random() < epsilon:
        return random.randint(0, num_actions - 1)  # explore
    else:
        with torch.no_grad():
            q_values = policy_net(*state)  # unpack pose, occupancy_patch, belief_patch, goal_id
            
            #print(f"Q-values: {q_values}")  # Debugging line
            return q_values.argmax().item()  # exploit

def update_target_network(policy_net, target_net):
    target_net.load_state_dict(policy_net.state_dict())

def compute_reward(action, previous_pose, new_pose, previous_poses_buffer, previous_cells_buffer, previous_belief_patch, next_belief_patch, target_class_id, found_target):

    # Initialize reward
    reward = 0.0

    # Success
    if found_target:
        return 10.0 

    # New pose processing
    if torch.equal(new_pose, previous_pose): # Invalid move (into wall)
        reward -= 1.0
    elif any(torch.equal(new_pose, p) for p in previous_poses_buffer):     # Repeated position
        reward -= 0.5
    elif new_pose[4:].tolist() in previous_cells_buffer:  # Repeated cell
        reward -= 0.25
    else: # Small bonus for exploring new pose
        reward += 0.5

    # Encourage moving forward
    if action == "move_forward":  
        reward += 0.1

    # Encourage high entropy in belief patch
    #previous_belief_entropy = compute_belief_patch_entropy(previous_belief_patch)
    #next_belief_entropy = compute_belief_patch_entropy(next_belief_patch)
    #reward += 100 * (previous_belief_entropy-next_belief_entropy)

    # Encourage high probability of target object in belief patch
    target_prob = compute_belief_patch_target_prob(next_belief_patch, target_class_id)
    previous_target_prob = compute_belief_patch_target_prob(previous_belief_patch, target_class_id)
    reward += 1000 * (target_prob - previous_target_prob)

    # Step cost
    reward -= 0.01

    return reward


# Helper functions for belief patch processing
def compute_belief_patch_entropy(belief_patch):
    """
    Computes the average entropy over valid spatial locations in the belief patch,
    where a location is considered valid if its class distribution is not all zero.

    Args:
        belief_patch (torch.Tensor): Tensor of shape (num_classes, height, width),
                                     representing class probabilities at each pixel.

    Returns:
        float: Mean entropy of the belief patch.
    """
    eps = 1e-10

    # Compute sum over classes to identify valid pixels (non-zero distributions)
    valid_mask = belief_patch.sum(dim=0) > 0  # Shape: (height, width)

    # Count valid pixels
    num_valid_pixels = valid_mask.sum().item()

    if num_valid_pixels == 0:
        return 0.0 # No valid pixels

    # Clamp to avoid log(0)
    belief_patch = belief_patch.clamp(min=eps)

    # Compute entropy at each pixel
    entropy_map = -torch.sum(belief_patch * torch.log(belief_patch), dim=0)  # Shape: (height, width)

    # Filter entropy to valid pixels
    valid_entropy = entropy_map[valid_mask]

    # Compute mean entropy
    mean_entropy = valid_entropy.mean().item()

    return mean_entropy

def compute_belief_patch_target_prob(belief_patch, target_class_id):
    """
    Computes the maximum probability of the target class in the belief patch.

    Args:
        belief_patch (torch.Tensor): Tensor of shape (num_classes, height, width),
                                     representing class probabilities at each pixel.
        target_class_id (int): The class ID for which to compute the probability.

    Returns:
        float: Maximum probability of the target class in the belief patch.
    """
    if not (0 <= target_class_id < belief_patch.shape[0]):
        raise ValueError(f"Invalid class ID {target_class_id}. Must be in range [0, {belief_patch.shape[0] - 1}]")

    target_prob = belief_patch[target_class_id].max().item()

    return target_prob
