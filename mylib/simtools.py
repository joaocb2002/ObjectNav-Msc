import habitat_sim
from scipy.spatial.transform import Rotation as R
from habitat.utils.visualizations import maps
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def display_sim_state(rgb_obs, depth_obs, topdown_map, occ_grid_map, agent_pos, agent_rot, agent_radius, pathfinder):
    """
    Displays a 4-panel visualization of the agent's simulation state, including:

    1. RGB observation from the agent's camera.
    2. Depth observation from the agent's depth sensor.
    3. Top-down map of the environment with agent position and orientation.
    4. Occupancy grid map with agent position and orientation.

    Agent's position is projected into map coordinates using a pathfinder. 
    Orientation is visualized using a direction arrow based on the agent's rotation (quaternion).

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

    # Top-down map
    ax3.imshow(topdown_map)
    ax3.set_title('topdown map')
    ax3.axis('off')
    topdown_resolution = [topdown_map.shape[1], topdown_map.shape[0]]

    # Occupancy grid
    ax4.imshow(occ_grid_map)
    ax4.set_title('occupancy grid')
    ax4.axis('off')
    occ_grid_resolution = [occ_grid_map.shape[1], occ_grid_map.shape[0]]

    # Black grid lines
    rows, cols = occ_grid_map.shape[:2]
    for i in range(rows):
        ax4.axhline(y=i-0.5, color='black', linewidth=0.5)
    for j in range(cols):
        ax4.axvline(x=j-0.5, color='black', linewidth=0.5)

    # Compute the agent position in the top-down map and occupancy grid
    map_x, map_y = maps.to_grid(agent_pos[0], agent_pos[2], topdown_resolution, pathfinder=pathfinder)
    grid_x, grid_y = maps.to_grid(agent_pos[0], agent_pos[2], occ_grid_resolution, pathfinder=pathfinder)

    # Compute agent yaw from quarternion
    r = R.from_quat([agent_rot.x, agent_rot.y, agent_rot.z, agent_rot.w])
    yaw = r.as_euler("xyz", degrees=False)[2]

    # Compute agent radius in both maps
    min_bounds, max_bounds = pathfinder.get_bounds()
    x_dim = max_bounds[0] - min_bounds[0]
    topdown_radius = int(agent_radius / x_dim * topdown_resolution[0])
    occ_grid_radius = int(agent_radius / x_dim * occ_grid_resolution[0])

    # Draw the agent position and orientation on the top-down map and occupancy grid
    ax3.add_patch(plt.Circle((map_x, map_y), topdown_radius, color="red", fill=True))
    #ax3.add_patch(plt.Arrow(map_x, map_y, topdown_radius * np.cos(yaw), topdown_radius * np.sin(yaw), width=topdown_radius / 2, color="blue"))
    ax4.add_patch(plt.Circle((grid_x, grid_y), occ_grid_radius, color="red", fill=True))
    #ax4.add_patch(plt.Arrow(grid_x, grid_y, occ_grid_radius * np.cos(yaw), occ_grid_radius * np.sin(yaw), width=occ_grid_radius / 2, color="blue"))

    plt.tight_layout()
    plt.show()