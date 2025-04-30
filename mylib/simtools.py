import habitat_sim
from scipy.spatial.transform import Rotation as R
from habitat.utils.visualizations import maps
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import json
import cv2

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
        class_id = int(cls[i][0])
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

def was_object_found(object_id, found_objects_ids, confidences, bboxes, threshold=0.5):
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
        class_id = int(found_objects_ids[i][0])
        confidence = confidences[i][0]
        if class_id == object_id and confidence >= threshold:
            return True, bboxes[i]

    return False, None

def compute_travelled_distance(start_pos, end_pos):
    """
    Compute the travelled distance between two positions in 3D space.

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




### THESE ARE FOR THE RANDOM ALGORITHM ###
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
    elif (action == "move_forward" and agent_rot == 270) or (action == "move_backward" and agent_rot == 90):
        return is_position_valid([agent_pos[0], agent_pos[1]+1], grid_occ_positions)
    elif (action == "move_forward" and agent_rot == 90) or (action == "move_backward" and agent_rot == 270):
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
    elif action == "turn_left":
        return agent_pos, (agent_rot + 90) % 360
    elif action == "turn_right":
        return agent_pos, (agent_rot - 90) % 360

    return None, None
