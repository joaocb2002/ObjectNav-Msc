import cv2
import habitat_sim
from scipy.spatial.transform import Rotation as R
from habitat.utils.visualizations import maps
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import json
from sklearn.cluster import KMeans
import heapq

# -----------------------------
# Display Functions
# -----------------------------
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
    ax1.set_title('rgb')
    ax1.axis('off')

    # Visualize the observations: Depth
    depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")
    ax2.imshow(depth_img)
    ax2.set_title('depth')
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

def display_topdown_maps(topdown_map, occ_grid_map, agent_positions_tpl, agent_radius_tpl, agent_yaw):
    """
    Displays the top-down map and occupancy grid map with the agent's position and orientation.

    Args:
        topdown_map (np.ndarray): Rendered top-down map image.
        occ_grid_map (np.ndarray): Rendered occupancy grid map image.
        agent_positions_tpl (tuple): Tuple containing the agent's position in the top-down map and occupancy grid.
        agent_radius_tpl (tuple): Tuple containing the agent's radius in the top-down map and occupancy grid.
        agent_yaw (float): Agent's yaw angle in degrees.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))

    # Compute the agent position and radius in the top-down map and occupancy grid
    (map_x, map_y), (grid_x, grid_y) =  agent_positions_tpl
    topdown_radius, occ_grid_radius = agent_radius_tpl

    # Top-down map
    ax1.imshow(topdown_map)
    ax1.set_title('topdown map (Z, X): [{:.0f}, {:.0f}]'.format(map_x, map_y))
    ax1.axis('off')

    # Occupancy grid
    ax2.imshow(occ_grid_map)
    ax2.set_title('occupancy grid (Z, X): [{:.0f}, {:.0f}]'.format(grid_x, grid_y))
    ax2.axis('off')

    # Black grid lines
    rows, cols = occ_grid_map.shape[:2]
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

def display_topdown_maps_with_target(topdown_map, occ_grid_map, agent_positions_tpl, agent_radius_tpl, agent_yaw, target_coords_tpl):
    """
    Displays the top-down map and occupancy grid map with the agent's position, orientation, and target position.

    Args:
        topdown_map (np.ndarray): Rendered top-down map image.
        occ_grid_map (np.ndarray): Rendered occupancy grid map image.
        agent_positions_tpl (tuple): Tuple containing the agent's position in the top-down map and occupancy grid.
        agent_radius_tpl (tuple): Tuple containing the agent's radius in the top-down map and occupancy grid.
        agent_yaw (float): Agent's yaw angle in degrees.
        target_coords_tpl (tuple): Tuple containing the target position in the top-down map and occupancy grid.

    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))

    # Compute the agent position and radius in the top-down map and occupancy grid
    (map_x, map_y), (grid_x, grid_y) =  agent_positions_tpl
    topdown_radius, occ_grid_radius = agent_radius_tpl
    real_target_pos, computed_target_pos = target_coords_tpl

    # Top-down map
    ax1.imshow(topdown_map)
    ax1.set_title('topdown map (Z, X): [{:.0f}, {:.0f}]'.format(map_x, map_y))
    ax1.axis('off')

    # Occupancy grid
    ax2.imshow(occ_grid_map)
    ax2.set_title('occupancy grid (Z, X): [{:.0f}, {:.0f}]'.format(grid_x, grid_y))
    ax2.axis('off')

    # Black grid lines
    rows, cols = occ_grid_map.shape[:2]
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
    target_pos_map, target_pos_grid = real_target_pos
    ax1.plot([target_pos_map[1] - topdown_radius*2/3, target_pos_map[1] + topdown_radius*2/3],
        [target_pos_map[0] + topdown_radius*2/3, target_pos_map[0] - topdown_radius*2/3], color='blue')
    ax1.plot([target_pos_map[1] - topdown_radius*2/3, target_pos_map[1] + topdown_radius*2/3],
        [target_pos_map[0] - topdown_radius*2/3, target_pos_map[0] + topdown_radius*2/3], color='blue')
    ax2.plot([target_pos_grid[1] - occ_grid_radius*2/3, target_pos_grid[1] + occ_grid_radius*2/3],
        [target_pos_grid[0] + occ_grid_radius*2/3, target_pos_grid[0] - occ_grid_radius*2/3], color='blue')
    ax2.plot([target_pos_grid[1] - occ_grid_radius*2/3, target_pos_grid[1] + occ_grid_radius*2/3],
        [target_pos_grid[0] - occ_grid_radius*2/3, target_pos_grid[0] + occ_grid_radius*2/3], color='blue')

    target_pos_map, target_pos_grid = computed_target_pos
    ax1.plot([target_pos_map[1] - topdown_radius*2/3, target_pos_map[1] + topdown_radius*2/3],
        [target_pos_map[0] + topdown_radius*2/3, target_pos_map[0] - topdown_radius*2/3], color='green')
    ax1.plot([target_pos_map[1] - topdown_radius*2/3, target_pos_map[1] + topdown_radius*2/3],
        [target_pos_map[0] - topdown_radius*2/3, target_pos_map[0] + topdown_radius*2/3], color='green')
    ax2.plot([target_pos_grid[1] - occ_grid_radius*2/3, target_pos_grid[1] + occ_grid_radius*2/3],
        [target_pos_grid[0] + occ_grid_radius*2/3, target_pos_grid[0] - occ_grid_radius*2/3], color='green')
    ax2.plot([target_pos_grid[1] - occ_grid_radius*2/3, target_pos_grid[1] + occ_grid_radius*2/3],
        [target_pos_grid[0] - occ_grid_radius*2/3, target_pos_grid[0] + occ_grid_radius*2/3], color='green')
    
    


    plt.tight_layout()
    plt.show()

def display_topdown_maps_with_clusters(topdown_map, occ_grid_map, agent_positions_tpl, agent_radius_tpl, agent_yaw, cluster_map, cluster_num, cluster_centers):
    """
    Displays the top-down map and occupancy grid map with the agent's position and orientation.

    Args:
        topdown_map (np.ndarray): Rendered top-down map image.
        occ_grid_map (np.ndarray): Rendered occupancy grid map image.
        agent_positions_tpl (tuple): Tuple containing the agent's position in the top-down map and occupancy grid.
        agent_radius_tpl (tuple): Tuple containing the agent's radius in the top-down map and occupancy grid.
        agent_yaw (float): Agent's yaw angle in degrees.
        cluster_map (dict): Mapping from (x, y) tuple to cluster index.
        cluster_num (int): The number of clusters.
        cluster_centers (list): List of cluster centers.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))

    # Compute the agent position and radius in the top-down map and occupancy grid
    (map_x, map_y), (grid_x, grid_y) =  agent_positions_tpl
    topdown_radius, occ_grid_radius = agent_radius_tpl

    # Top-down map
    ax1.imshow(topdown_map)
    ax1.set_title('topdown map (Z, X): [{:.0f}, {:.0f}]'.format(map_x, map_y))
    ax1.axis('off')

    # Occupancy grid
    ax2.imshow(occ_grid_map)
    ax2.set_title('occupancy grid (Z, X): [{:.0f}, {:.0f}]'.format(grid_x, grid_y))
    ax2.axis('off')

    # Black grid lines
    rows, cols = occ_grid_map.shape[:2]
    for i in range(rows):
        ax2.axhline(y=i-0.5, color='black', linewidth=0.5)
    for j in range(cols):
        ax2.axvline(x=j-0.5, color='black', linewidth=0.5)

    # Plot each cluster with a different color in occupancy grid
    for i in range(cluster_num):
        cluster_coords = [coord for coord, label in cluster_map.items() if label == i]
        if cluster_coords:
            x_coords, y_coords = zip(*cluster_coords)
            ax2.scatter(y_coords, x_coords, label=f'Cluster {i}', alpha=0.4)

    # Plot cluster centers
    for i, center in enumerate(cluster_centers):
        ax2.scatter(center[1], center[0], marker='x', color='black', s=100, label=f'Center {i}')

    # Draw the agent position and orientation on the top-down map and occupancy grid
    agent_yaw = math.radians(agent_yaw)
    ax1.add_patch(plt.Circle((map_y, map_x), topdown_radius*2/3, color="red", fill=True))
    ax1.add_patch(plt.Arrow(map_y, map_x, -topdown_radius * np.sin(agent_yaw), -topdown_radius * np.cos(agent_yaw), width=topdown_radius / 2, color="black"))
    ax2.add_patch(plt.Circle((grid_y, grid_x), occ_grid_radius*2/3, color="red", fill=True))
    ax2.add_patch(plt.Arrow(grid_y, grid_x, -occ_grid_radius * np.sin(agent_yaw), -occ_grid_radius * np.cos(agent_yaw), width=occ_grid_radius / 2, color="black"))

    plt.tight_layout()
    plt.show()

def display_topdown_and_entropy_maps(topdown_map, occ_grid_map, entropy_map, agent_positions_tpl, agent_radius_tpl, agent_yaw):
    """
    Displays the top-down map and occupancy grid map with the agent's position and orientation. 
    Also displays the entropy map.

    Args:
        topdown_map (np.ndarray): Rendered top-down map image.
        occ_grid_map (np.ndarray): Rendered occupancy grid map image.
        entropy_map (np.ndarray): 2D array representing the entropy map.
        agent_positions_tpl (tuple): Tuple containing the agent's position in the top-down map and occupancy grid.
        agent_radius_tpl (tuple): Tuple containing the agent's radius in the top-down map and occupancy grid.
        agent_yaw (float): Agent's yaw angle in degrees.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 3))

    # Compute the agent position and radius in the top-down map and occupancy grid
    (map_x, map_y), (grid_x, grid_y) =  agent_positions_tpl
    topdown_radius, occ_grid_radius = agent_radius_tpl

    # Top-down map
    ax1.imshow(topdown_map)
    ax1.set_title('topdown map (Z, X): [{:.0f}, {:.0f}]'.format(map_x, map_y))
    ax1.axis('off')

    # Occupancy grid
    ax2.imshow(occ_grid_map)
    ax2.set_title('occupancy grid (Z, X): [{:.0f}, {:.0f}]'.format(grid_x, grid_y))
    ax2.axis('off')

    # Entropy map
    ax3.imshow(entropy_map, cmap='hot')
    ax3.set_title('entropy map')
    ax3.axis('off')

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

    intrinsics_file = "camera_intrinsics.json"
    with open(intrinsics_file, "w") as f:
        json.dump(intrinsics, f, indent=4)

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

def merge_rgb_yolo_outputs(rgb, xyxy, cls, conf, names):
    """
    Merge YOLO outputs with RGB image. In place modification of the RGB image.
    Args:
        rgb (np.ndarray): RGB image.
        xyxy (np.ndarray): Bounding box coordinates.
        cls (np.ndarray): Class IDs.
        conf (np.ndarray): Confidence scores.
        names (list): List of class names.
    """
    for i in range(len(xyxy)):
        # Extract data
        box = xyxy[i].astype(int)
        class_id = int(cls[i])
        confidence = conf[i][0]
        label = f"{names[class_id]} {confidence:.2f}"

        # Get color for the class
        color = get_class_color(class_id)

        # Draw rectangle
        cv2.rectangle(rgb, (box[0], box[1]), (box[2], box[3]), color, 2)

        # Put label background
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        cv2.rectangle(rgb, 
                    (box[0], box[1] - text_height - baseline), 
                    (box[0] + text_width, box[1]), 
                    color, -1)

        # Text color
        text_color = (255, 255, 255, 255) if sum(color) < 382 else (0, 0, 0, 255)

        # Put label text
        cv2.putText(rgb, label, (box[0], box[1] - baseline), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 1)

def parse_detection_results(results):
    """
    Extract detection outputs from YOLO-like model results.

    Args:
        results: List of detection results (e.g., from Ultralytics YOLOv8).

    Returns:
        tuple: (xyxy, conf, cls, prob_vectors, names)
            - xyxy: ndarray of shape (N, 4), bounding boxes in [x1, y1, x2, y2] format
            - conf: ndarray of shape (N,), confidence scores
            - cls: ndarray of shape (N,), class IDs
            - prob_vectors: ndarray of shape (N, num_classes), per-class probabilities
            - names: list or dict of class names
            - num_detections: int, number of detections
    """
    result = results[0]  # One image, one result

    xyxy = result.boxes.xyxy.cpu().numpy()         # (N, 4)
    conf = result.boxes.conf.cpu().numpy()         # (N,)
    cls = result.boxes.cls.cpu().numpy()           # (N,)
    prob_vectors = result.boxes.data[:, 6:].cpu().numpy()  # (N, num_classes)
    names = result.names                           # class names
    num_detections = len(xyxy)                    # number of detections

    return xyxy, conf, cls, prob_vectors, names, num_detections


# -----------------------------
# Simulation Functions
# -----------------------------
def was_object_found(object_id, found_objects_ids, confidences, bboxes, threshold=0.80):
    """
    Check if an object was found based on its ID and confidence score and return its bounding box.
    Args:
        object_id (int): The ID of the object to check.
        found_objects_ids (list): List of found object IDs.
        confidences (list): List of confidence scores for the found objects.
        bboxes (list): List of bounding boxes for the found objects.
        threshold (float): Confidence threshold for considering an object as found.
    """
    for i in range(len(found_objects_ids)):
        class_id = int(found_objects_ids[i])
        confidence = confidences[i][0]
        if class_id == object_id and confidence >= threshold:
            return True, bboxes[i]

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

def compute_real_world_position(agent_pos, agent_rot, depth_obs, x_cam, y_cam, camera_intrinsics):
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

def compute_closeness(real_pos, computed_pos):
    """
    Compute the closeness between two positions in 2D space.
    Args:
        real_pos (list): Real-world position [x, y, z].
        computed_pos (list): Computed position [x, y, z].
    Returns:
        float: The Euclidean distance between the two positions, using only x and z coordinates.
    """
    real_pos_2d = np.array([real_pos[0], real_pos[2]])
    computed_pos_2d = np.array([computed_pos[0], computed_pos[2]])
    return np.linalg.norm(real_pos_2d - computed_pos_2d)

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
def cluster_mapping(occ_grid_map, cluster_num):
    """
    Cluster the occupancy grid map into a specified number of clusters using KMeans.

    Args:
        occ_grid_map (np.ndarray): An (N, 2) array where each row is (x, y) of a free cell.
        cluster_num (int): The number of clusters to form.

    Returns:
        dict: Mapping from (x, y) tuple to cluster index.
    """
    kmeans = KMeans(n_clusters=cluster_num, n_init=100, random_state=0)
    labels = kmeans.fit_predict(occ_grid_map)
    
    # Create mapping from coordinate tuple to cluster label
    cluster_map = {tuple(coord): label for coord, label in zip(occ_grid_map, labels)}
    
    return cluster_map

def get_cluster_centers(cluster_map, cluster_num):
    """
    Get the closest coordinate to the mean of each cluster.

    Args:
        cluster_map (dict): Mapping from (x, y) tuple to cluster index.
        cluster_num (int): The number of clusters.

    Returns:
        list: List of cluster centers.
    """
    cluster_centers = []

    for i in range(cluster_num):
        coords = [np.array(coord) for coord, label in cluster_map.items() if label == i]
        if coords:
            coords_array = np.vstack(coords)
            center = np.mean(coords_array, axis=0)
            distances = np.linalg.norm(coords_array - center, axis=1)
            closest_coord = list(coords_array[np.argmin(distances)])
            cluster_centers.append(closest_coord)

    return cluster_centers

def get_closest_cluster_center_navmesh(agent_pos, world_cluster_centers, pathfinder):
    """
    Get the closest cluster center to the agent's position.

    Args:
        agent_pos (list): Agent's position [x, y, z].
        world_cluster_centers (list): List of cluster centers in world coordinates [x, y, z].
        pathfinder (habitat_sim.PathFinder): PathFinder object for navigation.

    Returns:
        list: Closest world cluster center [x, y, z].
        float: Distance to the closest cluster center.
    """
    closest_center = None
    min_distance = float('inf')

    for center in world_cluster_centers:

        # Create a ShortestPath object
        shortest_path = habitat_sim.nav.ShortestPath()

        # Set start and end positions (numpy arrays of 3D coordinates)
        shortest_path.requested_start = np.array(agent_pos)
        shortest_path.requested_end = np.array(center)

        # Use find_path
        found = pathfinder.find_path(shortest_path)

        # Check result
        if found:
            distance = shortest_path.geodesic_distance
            if distance < min_distance:
                min_distance = distance
                closest_center = center

    return closest_center, min_distance

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



# -----------------------------
# A* Functions
# -----------------------------
def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def a_star(free_cells, start, goal):
    start = tuple(start)
    goal = tuple(goal)

    # Convert free cell list to set of tuples for O(1) lookup
    free_set = {tuple(cell) for cell in free_cells}

    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), 0, start, [start]))
    visited = set()

    while open_set:
        est_total, cost_so_far, current, path = heapq.heappop(open_set)
        current = tuple(current)

        if current == goal:
            return path
        if current in visited:
            continue
        visited.add(current)

        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if neighbor in free_set and neighbor not in visited:
                heapq.heappush(
                    open_set,
                    (
                        cost_so_far + 1 + heuristic(neighbor, goal),
                        cost_so_far + 1,
                        neighbor,
                        path + [neighbor]
                    )
                )
    return None


# -----------------------------
# Path Planning Functions
# -----------------------------
def compute_facing_rotation(agent_pos, goal_pos):
    """
    Returns one of [0, 90, 180, 270], representing the best orientation
    for the agent to face the goal cell.
    """
    dx = goal_pos[0] - agent_pos[0]
    dy = goal_pos[1] - agent_pos[1]

    # Determine the dominant axis
    if abs(dx) > abs(dy):
        return 180 if dx > 0 else 0   # Down or Up
    else:
        return 270 if dy > 0 else 90  # Right or Left
    
def compute_rotation_action(current_rotation, desired_rotation):
    """
    Returns one of ['turn_left', 'turn_right', 'turn_around', None]
    to rotate from current_rotation to desired_rotation.
    """
    delta = (desired_rotation - current_rotation) % 360

    if delta == 0:
        return None
    elif delta == 90:
        return 'turn_left'
    elif delta == 180:
        return 'turn_around'
    elif delta == 270:
        return 'turn_right'
    else:
        raise ValueError("Invalid rotation difference")

def compute_move_action_relative(current_pos, next_pos, current_rotation):
    """
    Returns a relative move action from current_pos to next_pos
    based on current_rotation.
    
    Output is one of:
        'move_forward', 'move_left', 'move_right', 'move_backward'
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
    delta = (direction - current_rotation) % 360

    if delta == 0:
        return 'move_forward'
    elif delta == 90:
        return 'move_left'
    elif delta == 180:
        return 'move_backward'
    elif delta == 270:
        return 'move_right'
    else:
        raise ValueError("Unexpected rotation delta")


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
