import torch
import numpy as np

def init_pose_tensor(grid_position, grid_resolution, agent_yaw, device='cuda'):
    """
    Initializes pose as a GPU tensor with normalized values.

    Args:
        grid_position (tuple or list of 2 floats/ints): x, y grid coordinates.
        grid_resolution (tuple or list of 2 floats/ints): x, y grid resolution.
        agent_yaw (float): Agent yaw angle in degrees.
        device (str): Target device (default 'cuda').

    Returns:
        torch.Tensor: Pose tensor on the specified device.
    """
    pose_values = [
        grid_position[0] / grid_resolution[0],
        grid_position[1] / grid_resolution[1],
        agent_yaw / 360.0
    ]

    # Convert to tensor and ensure correct type
    pose = torch.tensor(pose_values, dtype=torch.float32, device=device)
    return pose

def init_occupancy_patch(grid_position, grid_resolution, grid_map, patch_size, device='cuda'):
    """
    Initializes an occupancy patch as a GPU tensor from a NumPy grid map.

    Args:
        grid_position (tuple of ints): (cx, cz) agent grid coordinates in the grid map.
        grid_resolution (tuple of ints): Size of the grid map as (width, height).
        grid_map (np.ndarray): Grid map array with shape (width, height, 1), where 255 indicates an obstacle.
        patch_size (int): Size of the square patch to extract.
        device (str): Target device for the returned tensor.

    Returns:
        torch.Tensor: Occupancy patch of shape (patch_size, patch_size) on the specified device.
                      Occupied cells (obstacles or out-of-bounds) are set to 0.0; free cells are set to 1.0.
    """
    cx, cz = grid_position
    half_patch = patch_size // 2

    occ_patch = np.ones((patch_size, patch_size), dtype=np.float32)

    for dx in range(-half_patch, half_patch + 1):
        for dz in range(-half_patch, half_patch + 1):
            gx = cx + dx
            gz = cz + dz
            px = dx + half_patch
            pz = dz + half_patch

            if 0 <= gx < grid_resolution[0] and 0 <= gz < grid_resolution[1]:
                if grid_map[gx, gz, 0] == 255:
                    occ_patch[px, pz] = 0.0

    # Convert to tensor and ensure correct type
    occ_patch = torch.from_numpy(occ_patch).to(device)
    occ_patch = occ_patch.unsqueeze(0)  # Add channel dimensions [1]

    return occ_patch

def init_belief_patch(grid_position, grid_resolution, grid_map, belief_map, 
                      patch_size, num_classes, device='cuda'):
    """
    Initializes a belief patch as a GPU tensor from a nested Python list belief map.

    Args:
        grid_position (tuple of ints): (cx, cz) agent grid coordinates in the grid map.
        grid_resolution (tuple of ints): Size of the grid map as (width, height).
        grid_map (np.ndarray): Grid map array with shape (width, height, 1), where 255 indicates an obstacle.
        belief_map (list of list of np.ndarray): Belief map structured as a nested Python list 
            with shape [width][height] and each element being a NumPy array of shape (num_classes,).
        patch_size (int): Size of the square patch to extract.
        num_classes (int): Number of classes (matches belief_map element size).
        device (str): Target device for the returned tensor.

    Returns:
        torch.Tensor: Belief patch of shape (patch_size, patch_size, num_classes) on the specified device.
                      Cells corresponding to obstacles or out-of-bounds areas are filled with zeros.
    """
    cx, cz = grid_position
    half_patch = patch_size // 2

    belief_patch = np.ones((patch_size, patch_size, num_classes), dtype=np.float32)
    all_zeros = np.zeros((num_classes,), dtype=np.float32)

    for dx in range(-half_patch, half_patch + 1):
        for dz in range(-half_patch, half_patch + 1):
            gx = cx + dx
            gz = cz + dz
            px = dx + half_patch
            pz = dz + half_patch

            if 0 <= gx < grid_resolution[0] and 0 <= gz < grid_resolution[1]:
                if grid_map[gx, gz, 0] != 255:
                    belief_patch[px, pz] = belief_map[gx][gz]

    belief_patch = torch.from_numpy(belief_patch).to(device)

    return belief_patch.permute(2, 0, 1)  # Change to (num_classes, patch_size, patch_size)

def batch_single_sample(pose, occupancy_patch, belief_patch, goal, hidden_state=None):
    """
    Batch a single sample for DQN input.

    Args:
        pose (torch.Tensor): Current pose tensor.
        occupancy_patch (torch.Tensor): Occupancy patch tensor.
        belief_patch (torch.Tensor): Belief patch tensor.
        goal (torch.Tensor): Goal tensor.
        num_actions (int): Number of possible actions.
        hidden_state (torch.Tensor, optional): Hidden state for LSTM.

    Returns:
        tuple: Batched tensors ready for DQN input.
    """
    return (pose.unsqueeze(0), occupancy_patch.unsqueeze(0), belief_patch.unsqueeze(0),
            goal, hidden_state)

def unbatch_single_sample(batch_pose, batch_occupancy_patch, batch_belief_patch, batch_goal, batch_hidden_state=None):
    """
    Unbatch a single sample from DQN input.

    Args:
        batch_pose (torch.Tensor): Batched pose tensor.
        batch_occupancy_patch (torch.Tensor): Batched occupancy patch tensor.
        batch_belief_patch (torch.Tensor): Batched belief patch tensor.
        batch_goal (torch.Tensor): Batched goal tensor.
        batch_hidden_state (torch.Tensor, optional): Batched hidden state for LSTM.

    Returns:
        tuple: Individual tensors for DQN input.
    """
    return (batch_pose.squeeze(0), batch_occupancy_patch.squeeze(0), batch_belief_patch.squeeze(0),
            batch_goal, batch_hidden_state)

