import torch
import numpy as np

def init_pose_tensor(grid_position, grid_resolution, agent_yaw, device='cuda'):
    """
    Initializes pose as a 6D tensor: [4D one-hot yaw, normalized x, normalized y].

    Args:
        grid_position (tuple): (x, y) grid coordinates of the agent.
        grid_resolution (tuple): (width, height) of the full grid map.
        agent_yaw (float or int): Yaw angle in degrees (should be one of [0, 90, 180, 270]).
        device (str): Target device (default 'cuda').

    Returns:
        torch.Tensor: (6,) tensor with one-hot yaw and normalized position.
    """

    # Normalize position
    x_norm = grid_position[0] / grid_resolution[0]
    y_norm = grid_position[1] / grid_resolution[1]

    # Convert yaw to 4D one-hot
    yaw_map = {0: 0, 90: 1, 180: 2, 270: 3}
    if agent_yaw not in yaw_map:
        raise ValueError(f"agent_yaw must be one of [0, 90, 180, 270], got {agent_yaw}")
    
    yaw_one_hot = [0.0] * 4
    yaw_one_hot[yaw_map[agent_yaw]] = 1.0

    # Combine into final pose tensor
    pose_values = yaw_one_hot + [x_norm, y_norm]
    pose_tensor = torch.tensor(pose_values, dtype=torch.float32, device=device)

    return pose_tensor

def init_occupancy_patch(grid_position, grid_resolution, grid_map, patch_size, device='cuda'):
    """
    Extracts a square occupancy patch around the agent's position.

    Args:
        grid_position (tuple of ints): (z, x) position in the grid map (i.e., row, column).
        grid_resolution (tuple of ints): (height, width) of the full grid map.
        grid_map (np.ndarray): Shape (height, width, 1), with 255 = free space.
        patch_size (int): Size of the square patch (must be odd).
        device (str): Target device for the output tensor.

    Returns:
        torch.Tensor: Tensor of shape (1, patch_size, patch_size) on the specified device.
                      Occupied/unknown cells = 1.0, free cells = 0.0.
    """
    if patch_size % 2 == 0:
        raise ValueError("patch_size must be odd")

    cz, cx = grid_position  # center of patch
    half = patch_size // 2
    height, width = grid_resolution

    occ_patch = np.ones((patch_size, patch_size), dtype=np.float32)  # default: occupied/unknown = 1.0

    for dx in range(-half, half + 1):
        for dz in range(-half, half + 1):
            gx = cx + dx
            gz = cz + dz
            px = dx + half
            pz = dz + half

            if 0 <= gx < width and 0 <= gz < height:
                # Assuming grid_map[gy, gx, 0] = 255 means free
                if grid_map[gz, gx, 0] == 255:  # free space
                    occ_patch[pz, px] = 0.0

    occ_tensor = torch.from_numpy(occ_patch).unsqueeze(0).to(device)  # shape: (1, H, W)
    return occ_tensor

def init_belief_patch(grid_position, grid_resolution, grid_map, belief_map, 
                      patch_size, num_classes, device='cuda'):
    """
    Initializes a belief patch as a GPU tensor from a nested list-based belief map.

    Args:
        grid_position (tuple of ints): (z, x) position in the grid map.
        grid_resolution (tuple of ints): (height, width) of the full map.
        grid_map (np.ndarray): (height, width, 1), where 255 = free space.
        belief_map (list of list of np.ndarray): belief_map[z][x] gives (num_classes,) belief vector.
        patch_size (int): Size of the square patch (must be odd).
        num_classes (int): Number of object classes.
        device (str): Device to place tensor on.

    Returns:
        torch.Tensor: Tensor of shape (num_classes, patch_size, patch_size) on the specified device.
                      Cells out of bounds or in free space are filled with zeros.
    """
    if patch_size % 2 == 0:
        raise ValueError("patch_size must be odd")

    cz, cx = grid_position  # center coordinates
    height, width = grid_resolution
    half = patch_size // 2

    # Initialize belief patch with zeros
    belief_patch = np.zeros((patch_size, patch_size, num_classes), dtype=np.float32)

    for dx in range(-half, half + 1):
        for dz in range(-half, half + 1):
            gx = cx + dx
            gz = cz + dz
            px = dx + half
            pz = dz + half

            if 0 <= gx < width and 0 <= gz < height:
                if grid_map[gz, gx, 0] != 255:  # not free space → could contain an object
                    belief_vec = belief_map[gz][gx]
                    if belief_vec.sum() > 0:
                        belief_patch[pz, px] = belief_vec / belief_vec.sum()  # normalize

    # Convert to tensor and permute to (C, H, W)
    belief_tensor = torch.from_numpy(belief_patch).permute(2, 0, 1).to(device)
    return belief_tensor

def init_target_id_vector(TARGET_OBJECT_CLASS_ID, num_classes=28, device='cuda'):
    """
    Creates a one-hot encoding of the target object class ID.

    Args:
        TARGET_OBJECT_CLASS_ID (int): Integer ID of the target object (0 to num_classes - 1).
        num_classes (int): Total number of object classes (default: 28).
        device (str): Target device (default: 'cuda').

    Returns:
        torch.Tensor: One-hot tensor of shape (num_classes,) on the specified device.
    """
    if not (0 <= TARGET_OBJECT_CLASS_ID < num_classes):
        raise ValueError(f"Invalid class ID {TARGET_OBJECT_CLASS_ID}. Must be in range [0, {num_classes - 1}]")

    one_hot = torch.zeros(num_classes, dtype=torch.float32, device=device)
    one_hot[TARGET_OBJECT_CLASS_ID] = 1.0
    return one_hot

def update_previous_poses_buffer(previous_poses_buffer, next_pose, max_size=10):
    """
    Updates the buffer of previous poses with the new pose.

    Args:
        previous_poses_buffer (list): List of previous poses.
        next_pose (torch.Tensor): New pose tensor to add.
        max_size (int): Maximum size of the buffer.

    Returns:
        list: Updated buffer containing the last `max_size` poses.
    """
    previous_poses_buffer.append(next_pose)
    if len(previous_poses_buffer) > max_size:
        previous_poses_buffer.pop(0)  # Keep only the last `max_size` poses
    return previous_poses_buffer

def update_previous_cells_buffer(previous_cells_buffer, next_cell, max_size=10):
    """
    Updates the buffer of previous cells with the new cell.

    Args:
        previous_cells_buffer (list): List of previous cells.
        next_cell (tuple): New cell coordinates to add (z, x).
        max_size (int): Maximum size of the buffer.

    Returns:
        list: Updated buffer containing the last `max_size` cells.
    """
    previous_cells_buffer.append(next_cell)
    if len(previous_cells_buffer) > max_size:
        previous_cells_buffer.pop(0)  # Keep only the last `max_size` cells
    return previous_cells_buffer

# def batch_single_sample(pose, occupancy_patch, belief_patch, goal, hidden_state=None):
#     """
#     Batch a single sample for DQN input.

#     Args:
#         pose (torch.Tensor): Current pose tensor.
#         occupancy_patch (torch.Tensor): Occupancy patch tensor.
#         belief_patch (torch.Tensor): Belief patch tensor.
#         goal (torch.Tensor): Goal tensor.
#         num_actions (int): Number of possible actions.
#         hidden_state (torch.Tensor, optional): Hidden state for LSTM.

#     Returns:
#         tuple: Batched tensors ready for DQN input.
#     """
#     return (pose.unsqueeze(0), occupancy_patch.unsqueeze(0), belief_patch.unsqueeze(0),
#             goal, hidden_state)

# def unbatch_single_sample(batch_pose, batch_occupancy_patch, batch_belief_patch, batch_goal, batch_hidden_state=None):
#     """
#     Unbatch a single sample from DQN input.

#     Args:
#         batch_pose (torch.Tensor): Batched pose tensor.
#         batch_occupancy_patch (torch.Tensor): Batched occupancy patch tensor.
#         batch_belief_patch (torch.Tensor): Batched belief patch tensor.
#         batch_goal (torch.Tensor): Batched goal tensor.
#         batch_hidden_state (torch.Tensor, optional): Batched hidden state for LSTM.

#     Returns:
#         tuple: Individual tensors for DQN input.
#     """
#     return (batch_pose.squeeze(0), batch_occupancy_patch.squeeze(0), batch_belief_patch.squeeze(0),
#             batch_goal, batch_hidden_state)

