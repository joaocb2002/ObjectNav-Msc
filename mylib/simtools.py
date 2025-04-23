import habitat_sim
from scipy.spatial.transform import Rotation as R
from habitat.utils.visualizations import maps
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import json

def display_sim_state(rgb_obs, depth_obs, topdown_map, occ_grid_map, agent_positions_tpl, agent_radius_tpl, agent_yaw):
    """
    Displays a 4-panel visualization of the agent's simulation state, including:

    1. RGB observation from the agent's camera.
    2. Depth observation from the agent's depth sensor.
    3. Top-down map of the environment with agent position and orientation.
    4. Occupancy grid map with agent position and orientation.

    Args:
        rgb_obs (np.ndarray): RGBA image from the agent's RGB sensor.
        depth_obs (np.ndarray): Depth image as a 2D array of float distances (in meters).
        topdown_map (np.ndarray): Rendered top-down map image.
        occ_grid_map (np.ndarray): Rendered occupancy grid map image.
        agent_positions_tpl (tuple): Tuple containing the agent's position in the top-down map and occupancy grid.
        agent_radius_tpl (tuple): Tuple containing the agent's radius in the top-down map and occupancy grid.
        agent_yaw (float): Agent's yaw angle in degrees.
    """
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 4))

    # Visualize the observations: RGB
    rgb_img = Image.fromarray(rgb_obs, mode="RGBA")
    ax1.imshow(rgb_img)
    ax1.set_title('rgb obs')
    ax1.axis('off')

    # Visualize the observations: Depth
    depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")
    ax2.imshow(depth_img)
    ax2.set_title('depth obs')
    ax2.axis('off')

    # Compute the agent position and radius in the top-down map and occupancy grid
    (map_x, map_y), (grid_x, grid_y) =  agent_positions_tpl
    topdown_radius, occ_grid_radius = agent_radius_tpl

    # Top-down map
    ax3.imshow(topdown_map)
    ax3.set_title('topdown map (Z, X): [{:.0f}, {:.0f}]'.format(map_x, map_y))
    ax3.axis('off')

    # Occupancy grid
    ax4.imshow(occ_grid_map)
    ax4.set_title('occupancy grid (Z, X): [{:.0f}, {:.0f}]'.format(grid_x, grid_y))
    ax4.axis('off')

    # Black grid lines
    rows, cols = occ_grid_map.shape[:2]
    for i in range(rows):
        ax4.axhline(y=i-0.5, color='black', linewidth=0.5)
    for j in range(cols):
        ax4.axvline(x=j-0.5, color='black', linewidth=0.5)

    # Draw the agent position and orientation on the top-down map and occupancy grid
    agent_yaw = math.radians(agent_yaw)
    ax3.add_patch(plt.Circle((map_y, map_x), topdown_radius*2/3, color="red", fill=True))
    ax3.add_patch(plt.Arrow(map_y, map_x, -topdown_radius * np.sin(agent_yaw), -topdown_radius * np.cos(agent_yaw), width=topdown_radius / 2, color="black"))
    ax4.add_patch(plt.Circle((grid_y, grid_x), occ_grid_radius*2/3, color="red", fill=True))
    ax4.add_patch(plt.Arrow(grid_y, grid_x, -occ_grid_radius * np.sin(agent_yaw), -occ_grid_radius * np.cos(agent_yaw), width=occ_grid_radius / 2, color="black"))

    plt.tight_layout()
    plt.show()


def save_rgb_camera_intrinsics(sensor_spec):
    height, width = sensor_spec.resolution
    hfov = float(sensor_spec.hfov)  # Convert to float, default is 90 degrees

    fx = (width / 2.0) / math.tan(math.radians(hfov) / 2.0)
    fy = fx  # assuming square pixels
    cx = width / 2.0 # in habitat-sim, the camera is centered
    cy = height / 2.0

    intrinsics = {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "width": width,
        "height": height,
        "hfov": hfov
    }

    intrinsics_file = "camera_intrinsics.json"
    with open(intrinsics_file, "w") as f:
        json.dump(intrinsics, f, indent=4)


### THESE ARE FOR THE BASELINE ALGORITHM ###
def is_position_valid(position, grid_occ_positions):
    """
    Check if the position is valid based on the occupancy grid.

    Args:
        position (tuple): The position to check.
        grid_occ_positions (list): List of valid positions in the grid.

    Returns:
        bool: True if the position is valid, False otherwise.
    """
    # Check if the position is within bounds and not occupied
    if position in grid_occ_positions:
        return True

    return False

def is_action_valid(action, agent_pos, agent_rot, grid_occ_positions):
    """
    Check if the action is valid based on the agent's position, rotation, and occupancy grid.

    Args:
        action (str): The action to check.
        agent_pos (tuple): The agent's current position in the environment.
        agent_rot (float): The agent's current rotation in the environment.
        grid_occ_positions (list): List of occupied positions in the grid.

    Returns:
        bool: True if the action is valid, False otherwise.
    """

    # If action is 'turn_left' or 'turn_right', it's always valid
    if action in ["turn_left", "turn_right"]:
        return True
    
    # If action is 'move_forward' or 'move_backward', check the occupancy grid
    if (action == "move_forward" and agent_rot == 0) or (action == "move_backward" and agent_rot == 180):
        return is_position_valid([agent_pos[0]-1, agent_pos[1]], grid_occ_positions)
    elif (action == "move_forward" and agent_rot == 180) or (action == "move_backward" and agent_rot == 0):
        return is_position_valid([agent_pos[0]+1, agent_pos[1]], grid_occ_positions)
    elif (action == "move_forward" and agent_rot == 90) or (action == "move_backward" and agent_rot == 270):
        return is_position_valid([agent_pos[0], agent_pos[1]+1], grid_occ_positions)
    elif (action == "move_forward" and agent_rot == 270) or (action == "move_backward" and agent_rot == 90):
        return is_position_valid([agent_pos[0], agent_pos[1]-1], grid_occ_positions)
    
    return False

def perform_action(action, agent_pos, agent_rot):
    """
    Perform the action and update the agent's position and rotation.

    Args:
        action (str): The action to perform.
        agent_pos (tuple): The agent's current position in the environment grid.
        agent_rot (float): The agent's current rotation in the environment.

    Returns:
        tuple: Updated position and rotation of the agent.
    """
    if action == "move_forward":
        if agent_rot == 0:
            return [agent_pos[0]-1, agent_pos[1]], agent_rot
        elif agent_rot == 180:
            return [agent_pos[0]+1, agent_pos[1]], agent_rot
        elif agent_rot == 90:
            return [agent_pos[0], agent_pos[1]+1], agent_rot
        elif agent_rot == 270:
            return [agent_pos[0], agent_pos[1]-1], agent_rot
    elif action == "move_backward":
        if agent_rot == 0:
            return [agent_pos[0]+1, agent_pos[1]], agent_rot
        elif agent_rot == 180:
            return [agent_pos[0]-1, agent_pos[1]], agent_rot
        elif agent_rot == 90:
            return [agent_pos[0], agent_pos[1]-1], agent_rot
        elif agent_rot == 270:
            return [agent_pos[0], agent_pos[1]+1], agent_rot
    elif action == "turn_left":
        return agent_pos, (agent_rot + 90) % 360
    elif action == "turn_right":
        return agent_pos, (agent_rot - 90) % 360

    return None, None
