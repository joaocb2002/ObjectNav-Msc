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

    return torch.tensor(pose_values, dtype=torch.float32, device=device)

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
            else:
                occ_patch[px, pz] = 0.0

    return torch.from_numpy(occ_patch).to(device)

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
                if grid_map[gx, gz, 0] == 255:
                    belief_patch[px, pz] = all_zeros
                else:
                    belief_patch[px, pz] = belief_map[gx][gz]
            else:
                belief_patch[px, pz] = all_zeros

    return torch.from_numpy(belief_patch).to(device)
