import habitat_sim
from scipy.spatial.transform import Rotation as R
from habitat.utils.visualizations import maps
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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
        agent_pos (tuple): Agent's 3D position in the environment (x, y, z).
        agent_rot (tuple): Agent's orientation as a quaternion (x, y, z, w).
        agent_radius (float): Agent's radius in world units (for visualization).
        pathfinder (object): An object used to convert world coordinates to grid coordinates.
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
    ax3.add_patch(plt.Circle((map_y, map_x), topdown_radius*2/3, color="red", fill=True))
    ax3.add_patch(plt.Arrow(map_y, map_x, topdown_radius * np.sin(agent_yaw), -topdown_radius * np.cos(agent_yaw), width=topdown_radius / 2, color="black"))
    ax4.add_patch(plt.Circle((grid_y, grid_x), occ_grid_radius*2/3, color="red", fill=True))
    ax4.add_patch(plt.Arrow(grid_y, grid_x, occ_grid_radius * np.sin(agent_yaw), -occ_grid_radius * np.cos(agent_yaw), width=occ_grid_radius / 2, color="black"))

    plt.tight_layout()
    plt.show()