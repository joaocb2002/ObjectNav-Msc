import cv2
import habitat_sim
import random
from scipy.spatial.transform import Rotation as R
from habitat.utils.visualizations import maps
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
import math
import json
from sklearn.cluster import KMeans
import heapq
from collections import deque
from skimage.draw import line

# -----------------------------
# Constants
# -----------------------------
#MAX_ENTROPY = 3.29584 # 27 classes
MAX_ENTROPY = 3.33220 # 27 classses + background
MIN_ENTROPY = 3.30

# -----------------------------
# Display Functions - Standard
# -----------------------------
def display_sim_state(rgb_obs, depth_obs, topdown_map, grid_map, agent_positions, agent_radius, agent_yaw):
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
        grid_map (np.ndarray): Rendered occupancy grid map image.
        agent_positions (tuple): Tuple containing the agent's position in the top-down map and occupancy grid.
        agent_radius (tuple): Tuple containing the agent's radius in the top-down map and occupancy grid.
        agent_yaw (float): Agent's yaw angle in degrees.
    """
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 4))

    # Visualize the observations: RGB
    rgb_img = Image.fromarray(rgb_obs, mode="RGBA")
    ax1.imshow(rgb_img)
    ax1.set_title('rgb')
    ax1.axis('off')

    # Visualize the observations: Depth
    depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")
    ax2.imshow(depth_img)
    ax2.set_title('depth')
    ax2.axis('off')

    # Compute the agent position and radius in the top-down map and occupancy grid
    (map_x, map_y), (grid_x, grid_y) =  agent_positions
    topdown_radius, occ_grid_radius = agent_radius

    # Top-down map
    ax3.imshow(topdown_map)
    ax3.set_title('topdown map (Z, X): [{:.0f}, {:.0f}]'.format(map_x, map_y))
    ax3.axis('off')

    # Occupancy grid
    ax4.imshow(grid_map)
    ax4.set_title('occupancy grid (Z, X): [{:.0f}, {:.0f}]'.format(grid_x, grid_y))
    ax4.axis('off')

    # Black grid lines
    rows, cols = grid_map.shape[:2]
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
    
def display_sim_observations(rgb_obs, depth_obs):
    """
    Displays the RGB and depth observations from the simulation.

    Args:
        rgb_obs (np.ndarray): RGBA image from the agent's RGB sensor.
        depth_obs (np.ndarray): Depth image as a 2D array of float distances (in meters).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))

    # Visualize the observations: RGB
    rgb_img = Image.fromarray(rgb_obs, mode="RGBA")
    ax1.imshow(rgb_img)
    ax1.set_title('rgb')
    ax1.axis('off')

    # Visualize the observations: Depth
    depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")
    ax2.imshow(depth_img)
    ax2.set_title('depth')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

def display_topdown_maps(topdown_map, grid_map, agent_positions, agent_radius, agent_yaw):
    """
    Displays the top-down map and occupancy grid map with the agent's position and orientation.

    Args
        topdown_map (np.ndarray): Rendered top-down map image.
        grid_map (np.ndarray): Rendered occupancy grid map image.
        agent_positions (tuple): Tuple containing the agent's position in the top-down map and occupancy grid.
        agent_radius (tuple): Tuple containing the agent's radius in the top-down map and occupancy grid.
        agent_yaw (float): Agent's yaw angle in degrees.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))

    # Compute the agent position and radius in the top-down map and occupancy grid
    (map_x, map_y), (grid_x, grid_y) =  agent_positions
    topdown_radius, occ_grid_radius = agent_radius

    # Top-down map
    ax1.imshow(topdown_map)
    ax1.set_title('topdown map (Z, X): [{:.0f}, {:.0f}]'.format(map_x, map_y))
    ax1.axis('off')

    # Occupancy grid
    ax2.imshow(grid_map)
    ax2.set_title('occupancy grid (Z, X): [{:.0f}, {:.0f}]'.format(grid_x, grid_y))
    ax2.axis('off')

    # Black grid lines
    rows, cols = grid_map.shape[:2]
    for i in range(rows):
        ax2.axhline(y=i-0.5, color='black', linewidth=0.5)
    for j in range(cols):
        ax2.axvline(x=j-0.5, color='black', linewidth=0.5)

    # Draw the agent position and orientation on the top-down map and occupancy grid
    agent_yaw = math.radians(agent_yaw)
    ax1.add_patch(plt.Circle((map_y, map_x), topdown_radius*2/3, color="red", fill=True))
    ax1.add_patch(plt.Arrow(map_y, map_x, -topdown_radius * np.sin(agent_yaw), -topdown_radius * np.cos(agent_yaw), width=topdown_radius / 2, color="black"))
    ax2.add_patch(plt.Circle((grid_y, grid_x), occ_grid_radius*2/3, color="red", fill=True))
    ax2.add_patch(plt.Arrow(grid_y, grid_x, -occ_grid_radius * np.sin(agent_yaw), -occ_grid_radius * np.cos(agent_yaw), width=occ_grid_radius / 2, color="black"))

    plt.tight_layout()
    plt.show()

def display_topdown_maps_with_target(topdown_map, grid_map, agent_positions, agent_radius, agent_yaw, target_positions, real_target_positions):
    """
    Displays the top-down map and occupancy grid map with the agent's position, orientation, and target position.

    Args:
        topdown_map (np.ndarray): Rendered top-down map image.
        grid_map (np.ndarray): Rendered occupancy grid map image.
        agent_positions (tuple): Tuple containing the agent's position in the top-down map and occupancy grid.
        agent_radius (tuple): Tuple containing the agent's radius in the top-down map and occupancy grid.
        agent_yaw (float): Agent's yaw angle in degrees.
        target_positions (tuple): Tuple containing the estimated target position in the top-down map and occupancy grid.
        real_target_positions (tuple): Tuple containing the real target position in the top-down map and occupancy grid.

    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))

    # Compute the agent position and radius in the top-down map and occupancy grid
    (map_x, map_y), (grid_x, grid_y) =  agent_positions
    topdown_radius, occ_grid_radius = agent_radius

    # Top-down map
    ax1.imshow(topdown_map)
    ax1.set_title('topdown map (Z, X): [{:.0f}, {:.0f}]'.format(map_x, map_y))
    ax1.axis('off')

    # Occupancy grid
    ax2.imshow(grid_map)
    ax2.set_title('occupancy grid (Z, X): [{:.0f}, {:.0f}]'.format(grid_x, grid_y))
    ax2.axis('off')

    # Black grid lines
    rows, cols = grid_map.shape[:2]
    for i in range(rows):
        ax2.axhline(y=i-0.5, color='black', linewidth=0.5)
    for j in range(cols):
        ax2.axvline(x=j-0.5, color='black', linewidth=0.5)

    # Draw the agent position and orientation on the top-down map and occupancy grid
    agent_yaw = math.radians(agent_yaw)
    ax1.add_patch(plt.Circle((map_y, map_x), topdown_radius*2/3, color="red", fill=True))
    ax1.add_patch(plt.Arrow(map_y, map_x, -topdown_radius * np.sin(agent_yaw), -topdown_radius * np.cos(agent_yaw), width=topdown_radius / 2, color="black"))
    ax2.add_patch(plt.Circle((grid_y, grid_x), occ_grid_radius*2/3, color="red", fill=True))
    ax2.add_patch(plt.Arrow(grid_y, grid_x, -occ_grid_radius * np.sin(agent_yaw), -occ_grid_radius * np.cos(agent_yaw), width=occ_grid_radius / 2, color="black"))
    
    # Draw the target position on the top-down map and occupancy grid: diagonal crosses
    target_pos_map, target_pos_grid = real_target_positions
    ax1.plot([target_pos_map[1] - topdown_radius*2/3, target_pos_map[1] + topdown_radius*2/3],
        [target_pos_map[0] + topdown_radius*2/3, target_pos_map[0] - topdown_radius*2/3], color='blue')
    ax1.plot([target_pos_map[1] - topdown_radius*2/3, target_pos_map[1] + topdown_radius*2/3],
        [target_pos_map[0] - topdown_radius*2/3, target_pos_map[0] + topdown_radius*2/3], color='blue', label='real location')
    ax2.plot([target_pos_grid[1] - occ_grid_radius*2/3, target_pos_grid[1] + occ_grid_radius*2/3],
        [target_pos_grid[0] + occ_grid_radius*2/3, target_pos_grid[0] - occ_grid_radius*2/3], color='blue')
    ax2.plot([target_pos_grid[1] - occ_grid_radius*2/3, target_pos_grid[1] + occ_grid_radius*2/3],
        [target_pos_grid[0] - occ_grid_radius*2/3, target_pos_grid[0] + occ_grid_radius*2/3], color='blue', label='real location')

    target_pos_map, target_pos_grid = target_positions
    ax1.plot([target_pos_map[1] - topdown_radius*2/3, target_pos_map[1] + topdown_radius*2/3],
        [target_pos_map[0] + topdown_radius*2/3, target_pos_map[0] - topdown_radius*2/3], color='orange')
    ax1.plot([target_pos_map[1] - topdown_radius*2/3, target_pos_map[1] + topdown_radius*2/3],
        [target_pos_map[0] - topdown_radius*2/3, target_pos_map[0] + topdown_radius*2/3], color='orange', label='estimated location')
    ax2.plot([target_pos_grid[1] - occ_grid_radius*2/3, target_pos_grid[1] + occ_grid_radius*2/3],
        [target_pos_grid[0] + occ_grid_radius*2/3, target_pos_grid[0] - occ_grid_radius*2/3], color='orange')
    ax2.plot([target_pos_grid[1] - occ_grid_radius*2/3, target_pos_grid[1] + occ_grid_radius*2/3],
        [target_pos_grid[0] - occ_grid_radius*2/3, target_pos_grid[0] + occ_grid_radius*2/3], color='orange', label='estimated location')
    
    # Final touches - small legend and layout
    ax2.legend(loc='lower right', fontsize='small')
    ax1.legend(loc='lower right', fontsize='small')
    plt.tight_layout()
    plt.show()


# -----------------------------
# Display Functions - Baseline
# -----------------------------
def display_topdown_maps_with_clusters(topdown_map, grid_map, agent_positions, agent_radius, agent_yaw, cluster_map, cluster_centers):
    """
    Displays the top-down map and occupancy grid map with the agent's position and orientation.

    Args:
        topdown_map (np.ndarray): Rendered top-down map image.
        grid_map (np.ndarray): Rendered occupancy grid map image.
        agent_positions (tuple): Tuple containing the agent's position in the top-down map and occupancy grid.
        agent_radius (tuple): Tuple containing the agent's radius in the top-down map and occupancy grid.
        agent_yaw (float): Agent's yaw angle in degrees.
        cluster_map (dict): Mapping from (x, y) tuple to cluster index.
        cluster_centers (list): List of cluster centers.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))

    # Compute the agent position and radius in the top-down map and occupancy grid
    (map_x, map_y), (grid_x, grid_y) =  agent_positions
    topdown_radius, occ_grid_radius = agent_radius

    # Top-down map
    ax1.imshow(topdown_map)
    ax1.set_title('topdown map (Z, X): [{:.0f}, {:.0f}]'.format(map_x, map_y))
    ax1.axis('off')

    # Occupancy grid
    ax2.imshow(grid_map)
    ax2.set_title('occupancy grid (Z, X): [{:.0f}, {:.0f}]'.format(grid_x, grid_y))
    ax2.axis('off')

    # Black grid lines
    rows, cols = grid_map.shape[:2]
    for i in range(rows):
        ax2.axhline(y=i-0.5, color='black', linewidth=0.5)
    for j in range(cols):
        ax2.axvline(x=j-0.5, color='black', linewidth=0.5)

    # Plot each cluster with a different color in occupancy grid
    for i in range(len(cluster_centers)):
        cluster_coords = [coord for coord, label in cluster_map.items() if label == i]
        if cluster_coords:
            x_coords, y_coords = zip(*cluster_coords)
            ax2.scatter(y_coords, x_coords, label=f'Cluster {i}', alpha=0.2)

    # Plot cluster centers
    for i, center in enumerate(cluster_centers):
        ax2.scatter(center[1], center[0], marker='x', color='black', s=100, label=f'Center {i}', alpha=0.5)

    # Draw the agent position and orientation on the top-down map and occupancy grid
    agent_yaw = math.radians(agent_yaw)
    ax1.add_patch(plt.Circle((map_y, map_x), topdown_radius*2/3, color="red", fill=True))
    ax1.add_patch(plt.Arrow(map_y, map_x, -topdown_radius * np.sin(agent_yaw), -topdown_radius * np.cos(agent_yaw), width=topdown_radius / 2, color="black"))
    ax2.add_patch(plt.Circle((grid_y, grid_x), occ_grid_radius*2/3, color="red", fill=True))
    ax2.add_patch(plt.Arrow(grid_y, grid_x, -occ_grid_radius * np.sin(agent_yaw), -occ_grid_radius * np.cos(agent_yaw), width=occ_grid_radius / 2, color="black"))

    plt.tight_layout()
    plt.show()


# -----------------------------
# Display Functions - Dirichlet
# -----------------------------
def display_topdown_and_entropy_maps(topdown_map, occ_grid_map, entropy_map, cluster_map, cluster_centers, agent_positions, agent_radius, agent_yaw):
    """
    Displays the top-down map and occupancy grid map with the agent's position and orientation. 
    Also displays the entropy map.

    Args:
        topdown_map (np.ndarray): Rendered top-down map image.
        occ_grid_map (np.ndarray): Rendered occupancy grid map image.
        entropy_map (np.ndarray): 2D array representing the entropy map.
        agent_positions (tuple): Tuple containing the agent's position in the top-down map and occupancy grid.
        cluster_map (dict): Mapping from (x, y) tuple to cluster index.
        cluster_centers (list): List of cluster centers.
        agent_radius (tuple): Tuple containing the agent's radius in the top-down map and occupancy grid.
        agent_yaw (float): Agent's yaw angle in degrees.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 3))

    # Compute the agent position and radius in the top-down map and occupancy grid
    (map_x, map_y), (grid_x, grid_y) =  agent_positions
    topdown_radius, occ_grid_radius = agent_radius

    # Top-down map
    ax1.imshow(topdown_map)
    ax1.set_title('topdown map (Z, X): [{:.0f}, {:.0f}]'.format(map_x, map_y))
    ax1.axis('off')

    # Occupancy grid
    ax2.imshow(occ_grid_map)
    ax2.set_title('occupancy grid (Z, X): [{:.0f}, {:.0f}]'.format(grid_x, grid_y))
    ax2.axis('off')

    # Plot each cluster with a different color in occupancy grid
    for i in range(len(cluster_centers)):
        cluster_coords = [coord for coord, label in cluster_map.items() if label == i]
        if cluster_coords:
            x_coords, y_coords = zip(*cluster_coords)
            ax2.scatter(y_coords, x_coords, label=f'Cluster {i}', alpha=0.2)

    # Plot cluster centers
    for i, center in enumerate(cluster_centers):
        ax2.scatter(center[1], center[0], marker='x', color='black', s=100, label=f'Center {i}', alpha=0.5)

    # Mask the zero entries
    masked_entropy = np.ma.masked_where(entropy_map == 0, entropy_map)

    # Modify 'Reds' to avoid starting at white
    reds = plt.cm.get_cmap('Reds', 256)
    new_colors = reds(np.linspace(0.2, 1, 256))  # Start at 0.2 to avoid white
    custom_cmap = mcolors.ListedColormap(new_colors)
    custom_cmap.set_bad(color='white')  # Keep masked values white

    # Compute maximum entropy value in the masked entropy map
    max_entropy_value = max(np.max(masked_entropy), MAX_ENTROPY)
    min_entropy_value = min(np.min(masked_entropy), MIN_ENTROPY)

    # Add a colorbar for the entropy scale
    img = ax3.imshow(masked_entropy, cmap=custom_cmap, interpolation='nearest', vmin=min_entropy_value, vmax=max_entropy_value)
    ax3.set_title('Entropy Map')
    ax3.axis('off')
    cbar = fig.colorbar(img, ax=ax3)
    cbar.set_label('Entropy')

    # Black grid lines
    rows, cols = occ_grid_map.shape[:2]
    for i in range(rows):
        ax2.axhline(y=i-0.5, color='black', linewidth=0.5)
        ax3.axhline(y=i-0.5, color='black', linewidth=0.5)
    for j in range(cols):
        ax2.axvline(x=j-0.5, color='black', linewidth=0.5)
        ax3.axvline(x=j-0.5, color='black', linewidth=0.5)

    # Draw the agent position and orientation on the top-down map and occupancy grid
    agent_yaw = math.radians(agent_yaw)
    ax1.add_patch(plt.Circle((map_y, map_x), topdown_radius*2/3, color="red", fill=True))
    ax1.add_patch(plt.Arrow(map_y, map_x, -topdown_radius * np.sin(agent_yaw), -topdown_radius * np.cos(agent_yaw), width=topdown_radius / 2, color="black"))
    ax2.add_patch(plt.Circle((grid_y, grid_x), occ_grid_radius*2/3, color="red", fill=True))
    ax2.add_patch(plt.Arrow(grid_y, grid_x, -occ_grid_radius * np.sin(agent_yaw), -occ_grid_radius * np.cos(agent_yaw), width=occ_grid_radius / 2, color="black"))

    plt.tight_layout()
    plt.show()


# -----------------------------
# Display Functions - DRL
# -----------------------------
def display_topdown_and_entropy_maps_only(topdown_map, occ_grid_map, entropy_map, agent_positions, agent_radius, agent_yaw):
    """
    Displays the top-down map and occupancy grid map with the agent's position and orientation. 
    Also displays the entropy map.

    Args:
        topdown_map (np.ndarray): Rendered top-down map image.
        occ_grid_map (np.ndarray): Rendered occupancy grid map image.
        entropy_map (np.ndarray): 2D array representing the entropy map.
        agent_positions (tuple): Tuple containing the agent's position in the top-down map and occupancy grid.
        agent_radius (tuple): Tuple containing the agent's radius in the top-down map and occupancy grid.
        agent_yaw (float): Agent's yaw angle in degrees.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 3))

    # Compute the agent position and radius in the top-down map and occupancy grid
    (map_x, map_y), (grid_x, grid_y) =  agent_positions
    topdown_radius, occ_grid_radius = agent_radius

    # Top-down map
    ax1.imshow(topdown_map)
    ax1.set_title('topdown map (Z, X): [{:.0f}, {:.0f}]'.format(map_x, map_y))
    ax1.axis('off')

    # Occupancy grid
    ax2.imshow(occ_grid_map)
    ax2.set_title('occupancy grid (Z, X): [{:.0f}, {:.0f}]'.format(grid_x, grid_y))
    ax2.axis('off')

    # Mask the zero entries
    masked_entropy = np.ma.masked_where(entropy_map == 0, entropy_map)

    # Modify 'Reds' to avoid starting at white
    reds = plt.cm.get_cmap('Reds', 256)
    new_colors = reds(np.linspace(0.2, 1, 256))  # Start at 0.2 to avoid white
    custom_cmap = mcolors.ListedColormap(new_colors)
    custom_cmap.set_bad(color='white')  # Keep masked values white

    # Compute maximum entropy value in the masked entropy map
    max_entropy_value = max(np.max(masked_entropy), MAX_ENTROPY)
    min_entropy_value = min(np.min(masked_entropy), MIN_ENTROPY)

    # Add a colorbar for the entropy scale
    img = ax3.imshow(masked_entropy, cmap=custom_cmap, interpolation='nearest', vmin=min_entropy_value, vmax=max_entropy_value)
    ax3.set_title('Entropy Map')
    ax3.axis('off')
    cbar = fig.colorbar(img, ax=ax3)
    cbar.set_label('Entropy')

    # Black grid lines
    rows, cols = occ_grid_map.shape[:2]
    for i in range(rows):
        ax2.axhline(y=i-0.5, color='black', linewidth=0.5)
        ax3.axhline(y=i-0.5, color='black', linewidth=0.5)
    for j in range(cols):
        ax2.axvline(x=j-0.5, color='black', linewidth=0.5)
        ax3.axvline(x=j-0.5, color='black', linewidth=0.5)

    # Draw the agent position and orientation on the top-down map and occupancy grid
    agent_yaw = math.radians(agent_yaw)
    ax1.add_patch(plt.Circle((map_y, map_x), topdown_radius*2/3, color="red", fill=True))
    ax1.add_patch(plt.Arrow(map_y, map_x, -topdown_radius * np.sin(agent_yaw), -topdown_radius * np.cos(agent_yaw), width=topdown_radius / 2, color="black"))
    ax2.add_patch(plt.Circle((grid_y, grid_x), occ_grid_radius*2/3, color="red", fill=True))
    ax2.add_patch(plt.Arrow(grid_y, grid_x, -occ_grid_radius * np.sin(agent_yaw), -occ_grid_radius * np.cos(agent_yaw), width=occ_grid_radius / 2, color="black"))

    plt.tight_layout()
    plt.show()

#------------------------------
# Display target prob heat map
# -----------------------------
def display_gridmap_with_clusters_and_target_prob_heatmap(grid_map, target_prob_heat_map, agent_positions, agent_radius, agent_yaw, cluster_map, cluster_centers):
    """
    Displays the top-down map and occupancy grid map with the agent's position and orientation.

    Args:
        target_prob_heat_map (np.ndarray): 2D array representing the target probability in each cell.
        grid_map (np.ndarray): Rendered occupancy grid map image.
        agent_positions (tuple): Tuple containing the agent's position in the top-down map and occupancy grid.
        agent_radius (tuple): Tuple containing the agent's radius in the top-down map and occupancy grid.
        agent_yaw (float): Agent's yaw angle in degrees.
        cluster_map (dict): Mapping from (x, y) tuple to cluster index.
        cluster_centers (list): List of cluster centers.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))

    # Compute the agent position and radius in the top-down map and occupancy grid
    _, (grid_x, grid_y) =  agent_positions
    _, occ_grid_radius = agent_radius

    # Top-down map
    ax1.imshow(grid_map)
    ax1.set_title('topdown map')
    ax1.axis('off')

    # Occupancy grid with clusters
    ax2.imshow(grid_map)
    ax2.set_title('occupancy grid')
    ax2.axis('off')

    # Black grid lines
    rows, cols = grid_map.shape[:2]
    for i in range(rows):
        ax2.axhline(y=i-0.5, color='black', linewidth=0.5)
        ax1.axhline(y=i-0.5, color='black', linewidth=0.5)
    for j in range(cols):
        ax2.axvline(x=j-0.5, color='black', linewidth=0.5)
        ax1.axvline(x=j-0.5, color='black', linewidth=0.5)

    # Plot each cluster with a different color in occupancy grid
    for i in range(len(cluster_centers)):
        cluster_coords = [coord for coord, label in cluster_map.items() if label == i]
        if cluster_coords:
            x_coords, y_coords = zip(*cluster_coords)
            ax2.scatter(y_coords, x_coords, label=f'Cluster {i}', alpha=0.2)

    # Plot cluster centers
    for i, center in enumerate(cluster_centers):
        ax2.scatter(center[1], center[0], marker='x', color='black', s=100, label=f'Center {i}', alpha=0.5)

    # Draw the agent position and orientation on the top-down map and occupancy grid
    agent_yaw = math.radians(agent_yaw)
    ax2.add_patch(plt.Circle((grid_y, grid_x), occ_grid_radius*2/3, color="red", fill=True))
    ax2.add_patch(plt.Arrow(grid_y, grid_x, -occ_grid_radius * np.sin(agent_yaw), -occ_grid_radius * np.cos(agent_yaw), width=occ_grid_radius / 2, color="black"))

    # Mask the zero entries
    masked_target_prob = np.ma.masked_where(target_prob_heat_map == 0, target_prob_heat_map)

    # Modify 'Reds' to avoid starting at white
    reds = plt.cm.get_cmap('Reds', 256)
    new_colors = reds(np.linspace(0.2, 1, 256))  # Start at 0.2 to avoid white
    custom_cmap = mcolors.ListedColormap(new_colors)
    custom_cmap.set_bad(color='white')  # Keep masked values white

    # Compute maximum probability value in the masked target probability map
    max_prob_value = np.max(masked_target_prob)
    min_prob_value = 0

    # Add a colorbar for the probability scale
    img = ax1.imshow(masked_target_prob, cmap=custom_cmap, interpolation='nearest', vmin=min_prob_value, vmax=max_prob_value)
    ax1.set_title('Target Probability Map')
    ax1.axis('off')
    cbar = fig.colorbar(img, ax=ax1)
    cbar.set_label('Probability')

    plt.tight_layout()
    plt.show()

# -----------------------------
# Belief Map Functions
# -----------------------------
def compute_entropy_map(belief_map, occ_grid_map, free_color=(255, 255, 255)):
    """
    Computes entropy for object (grey) cells only. Sets entropy to 0 for free (white) cells.

    Parameters:
        belief_map (list of list of np.ndarray): Dirichlet belief map
        occ_grid_map (np.ndarray): 3D occupancy map (H x W x 3), with RGB values
        free_color (tuple): RGB color representing free cells (default is white)

    Returns:
        np.ndarray: 2D entropy map
    """
    height = len(belief_map)
    width = len(belief_map[0])
    entropy_map = np.zeros((height, width))

    for y in range(height):
        for x in range(width):
            if tuple(occ_grid_map[y][x]) == free_color:
                entropy_map[y][x] = 0.0  # Free cell
            else:
                alpha = belief_map[y][x]
                alpha = np.clip(alpha, 1e-6, None)
                probs = alpha / np.sum(alpha)
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                entropy_map[y][x] = entropy

    return entropy_map

def compute_average_entropy(belief_map, grid_cells):
    """
    Computes the average entropy of a list of grid cells.

    Args:
        belief_map (list of list of np.ndarray): Dirichlet belief map.
        grid_cells (list): List of grid cells, where each cell is a numpy array representing the Dirichlet distribution.

    Returns:
        float: Average entropy across all grid cells.
    """
    total_entropy = 0.0
    count = 0

    for cell in grid_cells:
        x, y = cell
        if belief_map[x][y] is not None:
            alphas = belief_map[x][y]
            alphas = np.clip(alphas, 1e-6, None)  # Avoid division by zero
            probs = alphas / np.sum(alphas)
            entropy = -np.sum(probs * np.log(probs + 1e-10))  # Add small value to avoid log(0)
            total_entropy += entropy
            count += 1

    if count == 0:
        return 0.0
    
    return total_entropy / count

def compute_max_target_probability(belief_map, grid_cells, target_class_idx):
    """
    Computes the maximum probability of a target class across a list of grid cells.

    Args:
        belief_map (list of list of np.ndarray): Dirichlet belief map.
        grid_cells (list): List of grid cells, where each cell is a tuple (x, y).
        target_class_idx (int): Index of the target class in the Dirichlet distribution.

    Returns:
        float: Maximum probability of the target class across all grid cells.
    """
    max_prob = 0.0

    for cell in grid_cells:
        x, y = cell
        if belief_map[x][y] is not None:
            alphas = belief_map[x][y]
            alphas = np.clip(alphas, 1e-6, None)  # Avoid division by zero
            prob = alphas[target_class_idx] / np.sum(alphas)
            max_prob = max(max_prob, prob)

    return max_prob

def check_target_probability_in_entropy_map(belief_map, grid_cells, target_class_idx, threshold):
    """
    Checks if the target class probability exceeds a threshold in the belief map.

    Args:
        belief_map (list of list of np.ndarray): Dirichlet belief map.
        grid_cells (list): List of grid cells, where each cell is a tuple (x, y).
        target_class_idx (int): Index of the target class in the Dirichlet distribution.
        threshold (float): Pseudo-count threshold to check against.

    Returns:
        bool: True if any cell has a Dirichlet parameter for the target class above the threshold, False otherwise.
        list: List of the single cell that exceed the threshold by the largest margin.
    """
    max_alpha = -1.0
    max_cell = None
    for cell in grid_cells:
        x, y = cell
        if belief_map[x][y] is not None:
            alphas = belief_map[x][y]
            if alphas[target_class_idx] > max_alpha:
                max_alpha = alphas[target_class_idx]
                max_cell = cell
    if max_alpha > threshold:
        return True, max_cell
    else:
        return False, None

def compute_target_prob_map(belief_map, occ_grid_map, target_class_idx, free_color=(255, 255, 255)):
    """
    Computes the target probability map from the Dirichlet belief map.

    Args:
        belief_map (list of list of np.ndarray): Dirichlet belief map.
        target_class_idx (int): Index of the target class in the Dirichlet distribution.

    Returns:
        np.ndarray: 2D array representing the target probability map.
    """
    height = len(belief_map)
    width = len(belief_map[0])
    target_prob_map = np.zeros((height, width))

    for y in range(height):
        for x in range(width):
            if tuple(occ_grid_map[y][x]) == free_color:
                target_prob_map[y][x] = 0.0  # Free cell
            elif belief_map[y][x] is not None:
                alphas = belief_map[y][x]
                alphas = np.clip(alphas, 1e-6, None)  # Avoid division by zero
                prob = alphas[target_class_idx] / np.sum(alphas)
                target_prob_map[y][x] = prob
            else:
                target_prob_map[y][x] = 0.0

    return target_prob_map

# -----------------------------
# Sensor Functions
# -----------------------------
def save_rgb_camera_intrinsics(sensor_spec):
    """
    Save the RGB camera intrinsics to a JSON file.
    Args:
        sensor_spec (habitat_sim.SensorSpec): The sensor specification for the RGB camera.
    """
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

    intrinsics_file = "camera-intrinsics.json"
    with open(intrinsics_file, "w") as f:
        json.dump(intrinsics, f, indent=4)

def get_camera_pos_from_agent_pos(agent_pos, camera_height):
    """
    Computes the camera position based on the agent's position and the camera height.

    Args:
        agent_pos (list): Agent's position [x, y, z].
        camera_height (float): Height of the camera above the agent's position.

    Returns:
        list: Camera position [x, y, z].
    """
    return [agent_pos[0], agent_pos[1]+camera_height, agent_pos[2]]

# -----------------------------
# YOLO Utils Functions
# -----------------------------
def get_class_color(class_id):
    """
    Get a random color for the class ID.
    Args:
        class_id (int): Class ID.
    Returns:
        tuple: RGBA color.
    """
    np.random.seed(class_id) 
    color = np.random.randint(0, 255, size=3).tolist()
    return (int(color[0]), int(color[1]), int(color[2]), 255)  # RGBA

def merge_rgb_yolo_outputs(rgb, detections):
    """
    Merge YOLO detections with an RGB image by drawing boxes and labels in-place.

    Args:
        rgb (np.ndarray): RGB image.
        detections (List[Dict]): List of detection dicts with keys:
            'box', 'class_id', 'confidence', 'name', 'prob_vector'
    """
    for det in detections:
        box = det['box']
        class_id = det['class_id']
        confidence = det['confidence']
        label = f"{det['name']} {confidence:.2f}"

        color = get_class_color(class_id)

        # Draw bounding box
        cv2.rectangle(rgb, (box[0], box[1]), (box[2], box[3]), color, 2)

        # Label background
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        cv2.rectangle(rgb,
                      (box[0], box[1] - text_height - baseline),
                      (box[0] + text_width, box[1]),
                      color, -1)

        # Text color (white if background is dark, black if light)
        text_color = (255, 255, 255, 255) if sum(color) < 382 else (0, 0, 0, 255)

        # Label text
        cv2.putText(rgb, label, (box[0], box[1] - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 1)

def parse_yolo_detections(results):
    """
    Parses YOLO-like detection results into a list of structured detection dictionaries.

    Args:
        results: List of detection results from YOLO (e.g., Ultralytics YOLOv11).

    Returns:
        List[Dict]: A list where each element contains:
            - 'box': Bounding box as int ndarray [x1, y1, x2, y2]
            - 'class_id': Integer class ID
            - 'name': Class name
            - 'confidence': Detection confidence (float)
            - 'prob_vector': Class probability distribution (np.ndarray)
    """
    result = results[0]  # One image, one result

    xyxy = result.boxes.xyxy.cpu().numpy()         # (N, 4)
    conf = result.boxes.conf.cpu().numpy()         # (N,)
    cls = result.boxes.cls.cpu().numpy()           # (N,)
    prob_vectors = result.boxes.probs.cpu().numpy()  # (N, num_classes)
    names = result.names                           # class names
    num_detections = len(xyxy)                    # number of detections

    detections = []
    for i in range(len(xyxy)):
        detections.append({
            'box': xyxy[i].astype(int),
            'class_id': int(cls[i]),
            'name': names[int(cls[i])],
            'confidence': float(conf[i][0]),
            'prob_vector': prob_vectors[i]
        })

    return detections

def get_box_center(box):
    """
    Computes the integer coordinates of the center point of a bounding box.

    Args:
        box (Iterable[int]): Bounding box in (x1, y1, x2, y2) format,
                             where (x1, y1) is the top-left and (x2, y2) is the bottom-right corner.

    Returns:
        Tuple[int, int]: (center_x, center_y), the integer pixel coordinates of the box center.
    """
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

# -----------------------------
# Metrics Functions
# -----------------------------
def was_target_found(object_id, detections, threshold=0.80):
    """
    Check if the target object was found in the detection results.

    Args:
        object_id (int): The ID of the target object to check.
        detections (List[Dict]): List of detection dicts with keys:
            'class_id', 'confidence', 'box', etc.
        threshold (float): Confidence threshold to consider a detection valid.

    Returns:
        Tuple[bool, np.ndarray or None]: (True, bbox) if found, else (False, None)
    """
    for det in detections:
        class_id = det['class_id']
        confidence = det['confidence']
        if class_id == object_id and confidence >= threshold:
            return True, det['box']

    return False, None

def compute_travelled_distance(start_pos, end_pos):
    """
    Compute the linear travelled distance between two positions in 3D space.

    Args:
        start_pos (list): Starting position [x, y, z].
        end_pos (list): Ending position [x, y, z].

    Returns:
        float: The Euclidean distance between the two positions.
    """
    return np.linalg.norm(np.array(start_pos) - np.array(end_pos))

def compute_location_error(real_pos, computed_pos):
    """
    Compute the distance between two positions in 3D space.
    Args:
        real_pos (list): Real-world position [x, y, z].
        computed_pos (list): Computed position [x, y, z].
    Returns:
        float: The Euclidean distance between the two positions, using only x and z coordinates.
    """
    real_pos_2d = np.array([real_pos[0], real_pos[2]])
    computed_pos_2d = np.array([computed_pos[0], computed_pos[2]])
    return np.linalg.norm(real_pos_2d - computed_pos_2d)


# -----------------------------
# Real world - Camera Mapping Functions
# -----------------------------
def compute_real_world_position_from_pixel(agent_pos, agent_rot, depth_obs, x_cam, y_cam, camera_intrinsics):
    """
    Compute the real-world position of an object in the environment based on the agent's position, rotation, and depth and RGB observations.

    Args:
        agent_pos (list): Agent's position [x, y, z].
        agent_rot (list): Agent's rotation quaternion [x, y, z, w].
        depth_obs (float): Depth observation from the agent's depth sensor.
        x_cam (int): X coordinate of the pixel in the RGB image.
        y_cam (int): Y coordinate of the pixel in the RGB image.
        camera_intrinsics (dict): Camera intrinsics containing fx, fy, cx, cy.

    Returns:
        list: Real-world position [x, y, z] of the object.
    """
    # Acess camera intrinsics
    fx = camera_intrinsics["fx"]
    fy = camera_intrinsics["fy"]
    cx = camera_intrinsics["cx"]
    cy = camera_intrinsics["cy"]

    # Access quartenion
    q_w = agent_rot.w
    q_x = agent_rot.x
    q_y = agent_rot.y
    q_z = agent_rot.z

    # Compute real-world coordinates in camera coordinate system
    Z_c = -depth_obs
    X_c = (x_cam - cx) * depth_obs / fx
    Y_c = (y_cam - cy) * depth_obs / fy
    P_c = np.array([X_c, Y_c, Z_c]) 

    # Compute rotation matrix from quaternion
    Rot = R.from_quat([q_x, q_y, q_z, q_w]).as_matrix()

    # Compute real-world coordinates in world coordinate system
    P_w = np.dot(Rot, P_c) + np.array(agent_pos)
    
    return P_w

def compute_pixel_from_real_world_position(agent_pos, agent_rot, real_world_pos, camera_intrinsics):
    """
    Compute the pixel coordinates in the camera image from a real-world position.

    Args:
        agent_pos (list): Agent's position [x, y, z].
        agent_rot (list): Agent's rotation quaternion [x, y, z, w].
        real_world_pos (list or np.ndarray): Real-world position [x, y, z] of the object.
        camera_intrinsics (dict): Camera intrinsics containing fx, fy, cx, cy.

    Returns:
        tuple: Pixel coordinates (x_cam, y_cam) in the image.
    """
    # Unpack camera intrinsics
    fx = camera_intrinsics["fx"]
    fy = camera_intrinsics["fy"]
    cx = camera_intrinsics["cx"]
    cy = camera_intrinsics["cy"]

    # Unpack agent rotation quaternion
    q_w = agent_rot.w
    q_x = agent_rot.x
    q_y = agent_rot.y
    q_z = agent_rot.z

    # Compute inverse rotation matrix
    rot_mat = R.from_quat([q_x, q_y, q_z, q_w]).as_matrix()
    rot_inv = rot_mat.T  # Inverse of rotation matrix is its transpose

    # Convert real-world position to camera coordinate frame
    P_w = np.array(real_world_pos)
    T = np.array(agent_pos)
    P_c = np.dot(rot_inv, (P_w - T))
    X_c, Y_c, Z_c = P_c

    # Guard against division by zero
    if Z_c == 0:
        Z_c = 1e-6  # Small value to avoid division by zero

    # Project to image plane
    x_cam = int(round((X_c * fx) / -Z_c + cx))
    y_cam = int(round((Y_c * fy) / -Z_c + cy))

    return (x_cam, y_cam)


# -----------------------------
# Action Functions
# -----------------------------
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
    if action in ["turn_left", "turn_right", "turn_around"]:
        return True
    
    # If action is 'move_forward' or 'move_backward', check the occupancy grid
    if (action == "move_forward" and agent_rot == 0) or (action == "move_backward" and agent_rot == 180) or (action == "move_left" and agent_rot == 270) or (action == "move_right" and agent_rot == 90):
        return is_position_valid([agent_pos[0]-1, agent_pos[1]], grid_occ_positions)
    elif (action == "move_forward" and agent_rot == 180) or (action == "move_backward" and agent_rot == 0) or (action == "move_left" and agent_rot == 90) or (action == "move_right" and agent_rot == 270):
        return is_position_valid([agent_pos[0]+1, agent_pos[1]], grid_occ_positions)
    elif (action == "move_forward" and agent_rot == 270) or (action == "move_backward" and agent_rot == 90) or (action == "move_left" and agent_rot == 180) or (action == "move_right" and agent_rot == 0):
        return is_position_valid([agent_pos[0], agent_pos[1]+1], grid_occ_positions)
    elif (action == "move_forward" and agent_rot == 90) or (action == "move_backward" and agent_rot == 270) or (action == "move_left" and agent_rot == 0) or (action == "move_right" and agent_rot == 180):
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
        elif agent_rot == 270:
            return [agent_pos[0], agent_pos[1]+1], agent_rot
        elif agent_rot == 90:
            return [agent_pos[0], agent_pos[1]-1], agent_rot
    elif action == "move_backward":
        if agent_rot == 0:
            return [agent_pos[0]+1, agent_pos[1]], agent_rot
        elif agent_rot == 180:
            return [agent_pos[0]-1, agent_pos[1]], agent_rot
        elif agent_rot == 270:
            return [agent_pos[0], agent_pos[1]-1], agent_rot
        elif agent_rot == 90:
            return [agent_pos[0], agent_pos[1]+1], agent_rot
    elif action == "move_left":
        if agent_rot == 0:
            return [agent_pos[0], agent_pos[1]-1], agent_rot
        elif agent_rot == 180:
            return [agent_pos[0], agent_pos[1]+1], agent_rot
        elif agent_rot == 270:
            return [agent_pos[0]-1, agent_pos[1]], agent_rot
        elif agent_rot == 90:
            return [agent_pos[0]+1, agent_pos[1]], agent_rot
    elif action == "move_right":
        if agent_rot == 0:
            return [agent_pos[0], agent_pos[1]+1], agent_rot
        elif agent_rot == 180:
            return [agent_pos[0], agent_pos[1]-1], agent_rot
        elif agent_rot == 270:
            return [agent_pos[0]+1, agent_pos[1]], agent_rot
        elif agent_rot == 90:
            return [agent_pos[0]-1, agent_pos[1]], agent_rot
    elif action == "turn_around":
        return agent_pos, (agent_rot + 180) % 360
    elif action == "turn_left":
        return agent_pos, (agent_rot + 90) % 360
    elif action == "turn_right":
        return agent_pos, (agent_rot - 90) % 360

    return None, None


# -----------------------------
# Clustering Functions
# -----------------------------
def cluster_mapping(grid_map, cluster_num, random_state=42):
    """
    Cluster the occupancy grid map into a specified number of clusters using deterministic KMeans.

    Args:
        grid_map (np.ndarray): An (N, 2) array where each row is a free cell coordinate (x, y).
        cluster_num (int): The number of clusters to form.
        random_state (int): Seed for random number generator to ensure determinism.

    Returns:
        dict: Mapping from (x, y) tuple to cluster index.
    """
    kmeans = KMeans(
        n_clusters=cluster_num,
        n_init=1,                # only one initialization
        random_state=random_state  # fixed seed
    )
    labels = kmeans.fit_predict(grid_map)
    
    # Create mapping from coordinate tuple to cluster label
    cluster_map = {tuple(coord): label for coord, label in zip(grid_map, labels)}
    
    return cluster_map

def get_cluster_centers(cluster_map, cluster_num, top_n=1, seed=None):
    """
    Get a representative coordinate for each cluster, randomly selected from the top-N closest to the cluster mean.

    Args:
        cluster_map (dict): Mapping from (x, y) tuple to cluster index.
        cluster_num (int): The number of clusters.
        top_n (int): Number of closest candidates to consider for random choice.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        list: List of cluster center coordinates.
    """
    if seed is not None:
        random.seed(seed)

    cluster_centers = []

    for i in range(cluster_num):
        coords = [np.array(coord) for coord, label in cluster_map.items() if label == i]
        if coords:
            coords_array = np.vstack(coords)
            center = np.mean(coords_array, axis=0)
            distances = np.linalg.norm(coords_array - center, axis=1)

            # Get indices of the top-N closest points
            top_indices = np.argsort(distances)[:top_n]
            chosen_idx = random.choice(top_indices)
            cluster_centers.append(list(coords_array[chosen_idx]))

    return cluster_centers

def assign_cluster_centers_to_cells(grid_occ_cells, cluster_centers, grid_free_cells, max_dist=2):
    """
    Assigns each occupied cell to the nearest cluster center,
    only if it's within max_dist of a white (free) cell.

    Args:
        grid_occ_cells (list of list or tuple): List of occupied (gray) cells as [x, y] or (x, y).
        cluster_centers (list of list or tuple): List of cluster centers as [x, y] or (x, y).
        white_cells (list of list or tuple): List of free (white) cells as [x, y] or (x, y).
        max_dist (int): Maximum Manhattan distance to a white cell.

    Returns:
        dict: Mapping each cluster center to a list of eligible occupied cells assigned to it.
    """
    cluster_map = {tuple(center): [] for center in cluster_centers}
    grid_free_cells_set = set(tuple(cell) for cell in grid_free_cells)  # Convert to tuple for set

    for cell in grid_occ_cells:
        cell_tuple = tuple(cell)

        # Check if the cell is close enough to any white cell
        within_range = any(
            abs(cell_tuple[0] - white[0]) + abs(cell_tuple[1] - white[1]) <= max_dist
            for white in grid_free_cells_set
        )

        if not within_range:
            continue  # Discard gray cell too far from white space

        # Find nearest cluster center (Euclidean)
        closest_center = None
        min_distance = float('inf')
        for center in cluster_centers:
            distance = np.linalg.norm(np.array(cell_tuple) - np.array(center))
            if distance < min_distance:
                min_distance = distance
                closest_center = center

        if closest_center is not None:
            cluster_map[tuple(closest_center)].append(cell_tuple)

    return cluster_map

def num_cluster_centers(total_white_cells, alpha=0.3, beta=0.5, min_clusters=1, max_clusters=None):
    """
    Compute initial number of cluster centers for object search in a grid-based indoor space.

    Parameters:
    - total_white_cells (int): Number of white (navigable) cells in the environment.
    - alpha (float): Scaling factor for tuning cluster aggressiveness.
    - beta (float): Exponent controlling growth rate; default 0.5 gives sqrt scaling.
    - min_clusters (int): Minimum allowed cluster centers (default = 1).
    - max_clusters (int or None): Optional cap on maximum cluster centers.

    Returns:
    - int: Number of initial cluster centers.
    """
    if total_white_cells <= 0:
        return 0

    estimated = math.ceil(alpha * (total_white_cells ** beta))
    estimated = max(min_clusters, estimated)
    if max_clusters is not None:
        estimated = min(max_clusters, estimated)
    return estimated

# -----------------------------
# Path Planning Functions
# -----------------------------
def get_closest_cluster_path(agent_pos, grid_cluster_centers, grid_free_cells):
    """
    Get the path to the closest cluster center from the agent's position in 2D space.

    Args:
        agent_pos (tuple): Agent's position (x, y).
        grid_cluster_centers (list): List of cluster centers (x, y).
        grid_free_cells (2D list): Grid where 0 = free, 1 = occupied.

    Returns:
        tuple: Closest cluster center (x, y).
        list: Path to the closest cluster center.
    """
    agent_pos = tuple(agent_pos)
    grid_cluster_centers = [tuple(c) for c in grid_cluster_centers]

    closest_center = None
    min_path_length = float('inf')
    best_path = None

    for center in grid_cluster_centers:
        path = a_star(grid_free_cells, agent_pos, center)
        if path is not None and len(path) < min_path_length:
            closest_center = center
            min_path_length = len(path)
            best_path = path

    closest_center = list(closest_center) if closest_center is not None else None
    best_path = [list(p) for p in best_path] if best_path is not None else None

    # Remove first cell from the path
    if best_path is not None and len(best_path) > 0:
        best_path.pop(0)

    return closest_center, best_path

def compute_path(start, end, grid_free_cells):
    """
    Compute the path from start to end and its length using A* algorithm.

    Args:
        start (tuple): Starting position (x, y).
        end (tuple): Ending position (x, y).
        grid_free_cells (2D list): Grid where 0 = free, 1 = occupied.

    Returns:
        int: Length of the path.
        list: Path from start to end, or None if no path exists.
    """
    path = a_star(grid_free_cells, start, end)
    if not path:
        return float('inf'), None

    # Remove the first cell (start) from the path
    path = [list(p) for p in path[1:]]
    path_length = len(path)
    return path_length, path

def compute_relative_actions(current_pos, current_rot, next_pos, action_set):
    """
    Returns the shortest required set of actions to go from current_pos to next_pos
    based on current_rotation.

    Args:
        current_pos (tuple): Current position in the grid (x, y).
        current_rotation (int): Current rotation of the agent in degrees (0, 90, 180, 270).
        next_pos (tuple): Next position in the grid (x, y).
        action_set (list): List of valid actions.

    Returns:
        list: The set of actions to perform to move from current_pos to next_pos.
    """
    dx = next_pos[0] - current_pos[0]
    dy = next_pos[1] - current_pos[1]

    # Direction of movement in global terms
    direction = None
    if dx == -1 and dy == 0:
        direction = 0     # up
    elif dx == 0 and dy == -1:
        direction = 90    # left
    elif dx == 1 and dy == 0:
        direction = 180   # down
    elif dx == 0 and dy == 1:
        direction = 270   # right
    else:
        raise ValueError("Invalid move: cells are not adjacent")

    # Convert global direction to relative move based on current rotation
    delta = (direction - current_rot) % 360

    if delta == 0:
        return ['move_forward']
    elif delta == 90:
        if 'move_left' in action_set:
            return ['move_left']
        else:
            return ['turn_left', 'move_forward']
    elif delta == 270:
        if 'move_right' in action_set:
            return ['move_right']
        else:
            return ['turn_right', 'move_forward']
    elif delta == 180:
        if 'move_backward' in action_set:
            return ['move_backward']
        elif 'turn_around' in action_set:
            return ['turn_around', 'move_forward']
        else:
            return ['turn_left', 'turn_left', 'move_forward']
    else:
        raise ValueError("Unexpected rotation delta")

# -----------------------------
# A* Functions
# -----------------------------
def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def perpendicular_distance(point, line_start, line_end):
    # Return perpendicular distance from `point` to the line segment from `line_start` to `line_end`
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end
    dx, dy = x2 - x1, y2 - y1
    if dx == dy == 0:
        return 0  # Avoid division by zero if start==end
    num = abs(dy * px - dx * py + x2 * y1 - y2 * x1)
    denom = np.hypot(dx, dy)
    return num / denom

def a_star(free_cells, start, goal, alpha=0.1):
    start = tuple(start)
    goal = tuple(goal)

    free_set = {tuple(cell) for cell in free_cells}

    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), 0, start, [start]))
    visited = set()

    while open_set:
        est_total, cost_so_far, current, path = heapq.heappop(open_set)

        if current == goal:
            return path
        if current in visited:
            continue
        visited.add(current)

        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if neighbor in free_set and neighbor not in visited:
                g = cost_so_far + 1
                h = heuristic(neighbor, goal)
                line_penalty = alpha * perpendicular_distance(neighbor, start, goal)
                f = g + h + line_penalty
                heapq.heappush(open_set, (f, g, neighbor, path + [neighbor]))

    return None


# -----------------------------
# Map Functions
# -----------------------------
def get_2d_coords(object_position, topdown_resolution, occ_grid_resolution, pathfinder):
    """
    Convert a 3D object position into 2D grid coordinates for both top-down and occupancy grids.

    Args:
        object_position: tuple or list (x, y, z) in world coordinates
        topdown_resolution: resolution value for the top-down map
        occ_grid_resolution: resolution value for the occupancy grid
        pathfinder: Habitat pathfinder instance

    Returns:
        tuple: (map_position, grid_position)
            - map_position: 2D coordinate in top-down grid
            - grid_position: 2D coordinate in occupancy grid
    """
    x, y, z = object_position

    map_position = list(maps.to_grid(z, x, topdown_resolution, pathfinder=pathfinder))
    grid_position = list(maps.to_grid(z, x, occ_grid_resolution, pathfinder=pathfinder))

    return map_position, grid_position

def get_closest_grey_cell(white_cell, grid_occ_map):
	"""
	Find the closest grey cell to a given white cell in the occupancy grid, using Breadth-First Search (BFS).
	This function assumes that the white cell is free (255, 255, 255) and the grey cells are occupied (128, 128, 128).

	Args:
		white_cell (tuple): The coordinates of the white cell (x, y).
		grid_occ_map (np.ndarray): 2D occupancy grid map where (255, 255, 255) = free, (128, 128, 128) = occupied.

	Returns:
		tuple: Coordinates of the closest grey cell (x, y).
	"""

	# Directions for 4-connected neighbors
	directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

	# Initialize BFS queue and visited set
	queue = deque([white_cell])
	visited = set()
	visited.add(tuple(white_cell))

	while queue:
		current_cell = queue.popleft()

		# Check if the current cell is grey (occupied)
		if np.all(grid_occ_map[current_cell[0], current_cell[1]] == (128, 128, 128)):
			return current_cell

		# Explore neighbors
		for dx, dy in directions:
			neighbor = (current_cell[0] + dx, current_cell[1] + dy)

			# Check bounds and if the neighbor is already visited
			if (0 <= neighbor[0] < grid_occ_map.shape[0] and
				0 <= neighbor[1] < grid_occ_map.shape[1] and
				neighbor not in visited):
				visited.add(neighbor)
				queue.append(neighbor)

	return None  # No grey cell found


# -----------------------------
# Visibility Functions
# -----------------------------
def simulate_visibility_rays(grid_map, agent_grid_position, agent_yaw_deg, hfov_deg=90, num_rays=100, ray_length=None):
    """
    Casts visibility rays from the agent's position and orientation on the grid map, determining visible cells. 
    Each ray owns the closest cells it intersects, and rays are returned in index order. 
    
    Args:
        grid_map (np.ndarray): The occupancy grid map.
        agent_grid_position (tuple): The agent's position in the grid (z, x).
        agent_yaw_deg (float): The agent's yaw orientation in degrees.
        hfov_deg (float): Horizontal field of view in degrees.
        num_rays (int): Number of rays to cast.
        ray_length (float, optional): Length of each ray. If None, it will be computed based on the grid size.
    """
    # Agent position and radius
    grid_x, grid_y = agent_grid_position

    # Initialize variables
    Z, X = grid_map.shape[:2]
    rays = []
    ownership = {}

    # Angle calculations
    agent_yaw_rad = np.radians(agent_yaw_deg)
    hfov_rad = np.radians(hfov_deg)
    angles = np.linspace(agent_yaw_rad - hfov_rad / 2, agent_yaw_rad + hfov_rad / 2, num_rays)

    if ray_length is None:
        ray_length = np.hypot(Z, X)

    agent_zf, agent_xf = float(grid_x), float(grid_y)
    agent_z, agent_x = int(round(agent_zf)), int(round(agent_xf))

    # First pass: determine closest ray owner for each cell
    for i, angle in enumerate(angles):
        end_z = agent_zf - ray_length * np.cos(angle)
        end_x = agent_xf - ray_length * np.sin(angle)
        end_zi, end_xi = int(round(end_z)), int(round(end_x))

        rr, cc = line(agent_z, agent_x, end_zi, end_xi)

        for zi, xi in zip(rr, cc):
            if 0 <= zi < Z and 0 <= xi < X:
                cell = (zi, xi)
                dist = np.hypot(zi - agent_zf, xi - agent_xf)

                if cell not in ownership or dist < ownership[cell][1]:
                    ownership[cell] = (i, dist)
            else:
                break

    # Second pass: build per-ray list of owned cells
    rays = [[] for _ in range(num_rays)]

    for cell, (owner_idx, _) in ownership.items():
        rays[owner_idx].append(cell)

    return rays

def compute_visible_occ_cells(rays, grid_map, depth_map, grid_cells, world_positions, agent_grid_position, agent_rot, intrinsics, scene_height=3.0, agent_height=1.0):
    """
    Computes the visible occupancy cells from the agent's position and orientation in the grid map.

    Args:
        rays (list): List of rays, where each ray is a list of cells it owns.
        grid_map (np.ndarray): The occupancy grid map.
        grid_cells (list): List of grid cells in the format [[z, x], ...].
        world_positions (list): List of real-world positions corresponding to grid cells.
        agent_grid_position (tuple): The agent's position in the grid (z, x).
        agent_rot (float): The agent's rotation in degrees.
        intrinsics (dict): Camera intrinsics containing fx, fy, cx, cy 
        scene_height (float): Height of the scene in meters (default is 3.0).
        agent_height (float): Height of the agent in meters (default is 1.0).

    Returns:
        list: List of visible occupancy cells in the format [(z, x), ...].
    """

    # Initiallize the visible occupancy cells
    visible_occ_cells = []

    # Compute agent position in real world coordinates
    idx = grid_cells.index(list(agent_grid_position))
    agent_pos = world_positions[idx]

    # Draw visible cells (all in the same color)
    for ray in rays:
        for z, x in ray:
            # Find the index of the cell in grid_free_cells
            idx = grid_cells.index([z, x])
            real_world_position = world_positions[idx]
            
            # Compute distance from the agent to the cell in real world coordinates
            cell_pos = np.array(real_world_position)
            distance = np.linalg.norm(agent_pos - cell_pos)

            # Initialize depth value for the cell
            depth_values = []   

            # Verify depth for that cell
            vertical_step = np.linspace(0, scene_height, 20)

            for step in vertical_step:
                # Define the cell world position at the current height step
                cell_world_pos = np.array([real_world_position[0], step, real_world_position[2]])

                # Compute pixel from real world coordinates
                camera_pos = get_camera_pos_from_agent_pos(agent_pos, agent_height)
                pixel_x, pixel_y = compute_pixel_from_real_world_position(camera_pos, agent_rot, cell_world_pos, intrinsics)

                # Check if pixel is within bounds of the depth map
                if pixel_x < 0 or pixel_x >= depth_map.shape[1] or pixel_y < 0 or pixel_y >= depth_map.shape[0]:
                    continue  # Skip if pixel is out of bounds
                else:
                    depth_values.append(depth_map[pixel_y, pixel_x])

            # Compute the average depth value for the cell
            if depth_values:
                depth_value = np.mean(depth_values)
            else:
                depth_value = float('-Inf')

            if depth_value + 0.15 < distance or depth_value == float('-Inf'):
                break   

            # If the depth value is valid, add the cell to visible occupancy cells
            # Check if cell is occupied
            if tuple(grid_map[z, x]) != (255, 255, 255):
                visible_occ_cells.append((z, x))

    return visible_occ_cells

