# myutils.py
"""
Utility functions for processing maps and handling quaternions in the ObjectNav-Msc project.
"""
import cv2
import math
import quaternion
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, binary_dilation
from habitat.utils.visualizations import maps

### Map Processing Functions ###
def process_raw_topdown_map(raw_map):
    """
    Processes a raw occupancy grid map by converting it to RGB, adding axis arrows, and retaining the largest white chunk.
    
    Args:
        raw_map (np.array): 2D numpy array of shape (H, W) with values 0, 1, or 2.

    Returns:
        np.array: Processed map with the same shape as the input.
        tuple: Tuple containing the processed map and its first two dimensions (H, W).
    """

    processed_map = map_to_rgb(raw_map)
    processed_map = retain_largest_white_chunk(processed_map)
    processed_map = add_axis_to_map(processed_map)

    return processed_map, processed_map.shape[:2]

def process_raw_grid_map(raw_map, pathfinder):
    """
    Processes a raw occupancy grid map by converting it to RGB, adding axis arrows, and retaining the largest white chunk.
    
    Args:
        raw_map (np.array): 2D numpy array of shape (H, W) with values 0, 1, or 2.

    Returns:
        np.array: Processed map with the same shape as the input.
        tuple: Tuple containing the processed map and its first two dimensions (H, W).
    """

    processed_grid = map_to_rgb(raw_map)
    processed_grid = retain_largest_white_chunk(processed_grid)
    processed_grid = filter_grid_map(processed_grid, processed_grid.shape[:2], pathfinder)
    processed_grid = retain_largest_white_chunk(processed_grid)

    return processed_grid, processed_grid.shape[:2]

def find_free_cells(grid_map, grid_resolution, topdown_map, topdown_resolution, pathfinder):
    grid_free_cells, map_free_cells, world_free_coords = [], [], []

    for x_g in range(grid_resolution[0]):
        for y_g in range(grid_resolution[1]):
            if not is_white_pixel(grid_map, x_g, y_g):
                continue

            real_z, real_x = get_real_world_position(x_g, y_g, grid_resolution, pathfinder)
            map_x, map_y = get_topdown_position(real_x, real_z, topdown_resolution, pathfinder)

            if is_navigable_position(real_x, real_z, pathfinder) and is_white_pixel(topdown_map, map_x, map_y):
                grid_free_cells.append([x_g, y_g])
                map_free_cells.append([map_x, map_y])
                world_free_coords.append([real_x, 0.0, real_z])

    return grid_free_cells, map_free_cells, world_free_coords

def find_occupied_cells(grid_map, grid_resolution, topdown_map, topdown_resolution, pathfinder):
    grid_occ_cells, map_occ_cells, world_occ_coords = [], [], []

    for x_g in range(grid_resolution[0]):
        for y_g in range(grid_resolution[1]):

            real_z, real_x = get_real_world_position(x_g, y_g, grid_resolution, pathfinder)
            map_x, map_y = get_topdown_position(real_x, real_z, topdown_resolution, pathfinder)

            if not is_navigable_position(real_x, real_z, pathfinder) or not is_white_pixel(topdown_map, map_x, map_y) or not is_white_pixel(grid_map, x_g, y_g):
                grid_occ_cells.append([x_g, y_g])
                map_occ_cells.append([map_x, map_y])
                world_occ_coords.append([real_x, 0.0, real_z])

    return grid_occ_cells, map_occ_cells, world_occ_coords



### Secondary Map Processing Functions ###
def map_to_rgb(image):
    """
    Converts a 2D numpy array (H, W) where each element is a label (0, 1, or 2)
    into a 3D RGB image (H, W, 3) with the following mapping:
      - 0 -> [128, 128, 128] (gray)
      - 1 -> [256, 256, 256] (white)
      - 2 -> [0, 0, 0]       (black)
    
    Parameters:
        image (np.array): 2D numpy array of shape (H, W) with values 0, 1, or 2

    Returns:
        np.array: 3D numpy array of shape (H, W, 3) with RGB values
    """
    # Create an output image filled with zeros (black)
    h, w = image.shape
    rgb_image = np.zeros((h, w, 3), dtype=np.uint16)

    # Define the color map
    colormap = {
        0: [128, 128, 128],
        1: [255, 255, 255],
        2: [0, 0, 0]
    }

    for label, color in colormap.items():
        mask = image == label
        rgb_image[mask] = color

    return rgb_image

def add_axis_to_map(image, arrow_length_ratio=0.075):
    """
    Draws X and Z directional arrows from an origin point directly onto the image.

    Parameters:
    - image (np.array): Input image (grayscale or RGB).
    - origin (tuple): (x, y) coordinates for the origin point.
    - arrow_length_ratio (float): Fraction of image size to scale arrow length.
    
    Returns:
    - image_copy (np.array): Modified image with arrows and labels.
    """
    image_copy = image.copy()

    # Ensure RGB for drawing
    if len(image_copy.shape) == 2:
        image_copy = cv2.cvtColor(image_copy, cv2.COLOR_GRAY2BGR)

    height, width = image_copy.shape[:2]

    # Define arrow lengths based on image size
    arrow_len = int(min(height, width) * arrow_length_ratio)
    
    # Define origin point
    origin = (int(arrow_length_ratio * width/2), int(arrow_length_ratio * height/2))
    ox, oy = origin

    # Define endpoints of arrows
    x_end = (ox + arrow_len, oy)
    z_end = (ox, oy + arrow_len)

    # Draw arrows
    cv2.arrowedLine(image_copy, (ox, oy), x_end, color=(17, 17, 132), thickness=2, tipLength=0.15)
    cv2.arrowedLine(image_copy, (ox, oy), z_end, color=(17, 17, 132), thickness=2, tipLength=0.15)

    # Define font and size
    font_scale = arrow_len / 100.0
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2

    # Draw labels
    cv2.putText(image_copy, 'X', (int(x_end[0] + arrow_len / 3), int(x_end[1] + arrow_len / 5)), font, font_scale, (17, 17, 132), thickness)
    cv2.putText(image_copy, 'Z', (int(z_end[0] + arrow_len / 5), int(z_end[1] + arrow_len / 3)), font, font_scale, (17, 17, 132), thickness)

    return image_copy

def retain_largest_white_chunk(grid):
    """    
    Retains only the largest contiguous white chunk in a 3D RGB grid, converting all other
    white pixels to grey. The function also checks if the boundary of the chunk is enclosed
    and only retains chunks that are fully enclosed by black pixels.

    Parameters:
        grid (np.ndarray): 3D numpy array of shape (H, W, 3) representing the RGB grid.
    
    Returns:
        np.ndarray: Processed grid with only the largest white chunk retained as white,
                    and all other white pixels converted to grey.
    """
    white = np.array([255, 255, 255])
    black = np.array([0, 0, 0])
    grey = np.array([128, 128, 128])

    white_mask = np.all(grid == white, axis=-1)
    structure = np.array([[0,1,0],[1,1,1],[0,1,0]])

    labeled_array, num_features = label(white_mask, structure=structure)
    if num_features == 0:
        return grid

    # Preserve the original grid for boundary checks
    original_grid = grid.copy()

    counts = np.bincount(labeled_array.ravel())
    counts[0] = 0
    largest_label = np.argmax(counts)

    largest_chunk_mask = labeled_array == largest_label
    to_grey_mask = white_mask & ~largest_chunk_mask
    grid[to_grey_mask] = grey

    H, W = grid.shape[:2]
    for lbl in range(1, num_features + 1):
        if lbl == largest_label:
            continue

        chunk_mask = labeled_array == lbl
        if not np.any(chunk_mask):
            continue

        dilated = binary_dilation(chunk_mask, structure=structure)
        boundary_mask = dilated & ~chunk_mask

        # Skip if boundary touches edge (not enclosed)
        coords = np.argwhere(boundary_mask)
        if np.any((coords[:,0] == 0) | (coords[:,0] == H-1) |
                  (coords[:,1] == 0) | (coords[:,1] == W-1)):
            continue

        boundary_colors = original_grid[boundary_mask]
        if np.all(np.all(boundary_colors == black, axis=-1)):
            grid[boundary_mask] = grey

    return grid

def filter_grid_map(grid_map, grid_resolution, pathfinder):

    """
    Filters the grid map by marking non-navigable positions as grey.
   
    Parameters:
        grid_map (np.ndarray): 3D numpy array representing the grid map.
        grid_resolution (tuple): Resolution of the grid (height, width).
        pathfinder (habitat.PathFinder): Pathfinder object to check navigability.
    """
    for x_g in range(grid_resolution[0]):
        for y_g in range(grid_resolution[1]):
            if not is_white_pixel(grid_map, x_g, y_g):
                continue

            real_z, real_x = get_real_world_position(x_g, y_g, grid_resolution, pathfinder)
            
            if not is_navigable_position(real_x, real_z, pathfinder):
                mark_pixel_grey(grid_map, x_g, y_g)

    return grid_map


### Navigation Functions ###
def get_real_world_position(x_g, y_g, grid_resolution, pathfinder):
    """
    Converts grid coordinates (x_g, y_g) to real-world coordinates using the grid resolution
    and the pathfinder.

    Parameters:
        x_g (int): X coordinate in the grid.
        y_g (int): Y coordinate in the grid.
        grid_resolution (tuple): Resolution of the grid (height, width).
        pathfinder (habitat.PathFinder): Pathfinder object to convert grid to real-world coordinates.

    Returns:
        tuple: Real-world coordinates (real_z, real_x).
    """
    return maps.from_grid(x_g, y_g, grid_resolution, pathfinder=pathfinder)

def get_topdown_position(real_world_x, real_world_z, topdown_resolution, pathfinder):
    """
    Converts real-world coordinates (real_world_x, real_world_z) to top-down map coordinates
    using the top-down resolution and the pathfinder.

    Parameters:
        real_world_x (float): X coordinate in the real world.
        real_world_z (float): Z coordinate in the real world.
        topdown_resolution (tuple): Resolution of the top-down map (height, width).
        pathfinder (habitat.PathFinder): Pathfinder object to convert real-world to top-down coordinates.

    Returns:
        tuple: Top-down map coordinates (map_x, map_y).
    """
    return maps.to_grid(real_world_z, real_world_x, topdown_resolution, pathfinder=pathfinder)

def is_navigable_position(real_world_x, real_world_z, pathfinder):
    """
    Checks if a real-world position (real_world_x, real_world_z) is navigable
    using the pathfinder.

    Parameters:
        real_world_x (float): X coordinate in the real world.
        real_world_z (float): Z coordinate in the real world.
        pathfinder (habitat.PathFinder): Pathfinder object to check navigability.

    Returns:
        bool: True if the position is navigable, False otherwise.
    """
    return pathfinder.is_navigable([real_world_x, 0.0, real_world_z])


### Utility Functions ###
def is_white_pixel(grid_map, x, y):
    """
    Checks if the pixel at (x, y) in the grid map is white.

    Parameters:
        grid_map (np.ndarray): 3D numpy array representing the grid map.
        x (int): X coordinate of the pixel.
        y (int): Y coordinate of the pixel.
    Returns:
        bool: True if the pixel is white, False otherwise.
    """
    return grid_map[x, y, 0] == 255

def mark_pixel_grey(grid_map, x, y):
    """
    Marks the pixel at (x, y) in the grid map as grey.
    Parameters:
        grid_map (np.ndarray): 3D numpy array representing the grid map.
        x (int): X coordinate of the pixel.
        y (int): Y coordinate of the pixel.
    """
    grid_map[x, y, :] = [128, 128, 128]


### Bounding Box Scale Function ###
def compute_bbox_scale(bbox, rgb):
    """
    Computes the scale of a bounding box area relative to the RGB image area.

    Parameters:
        bbox (list or tuple): Bounding box coordinates in the format [x_min, y_min, x_max, y_max].
        rgb (np.ndarray): RGB image of shape (H, W, 3).
    Returns:
        float: Scale of the bounding box area relative to the RGB image area.
    """

    # Calculate the area of the bounding box
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    bbox_area = bbox_width * bbox_height

    # Calculate the area of the RGB image
    rgb_area = rgb.shape[0] * rgb.shape[1]

    # Compute the scale
    scale = 100*bbox_area / rgb_area

    return scale


### Quaternion and Yaw Functions ###
def quaternion_to_yaw(q):
    """
    Convert a quaternion.quaternion object to yaw angle in degrees.
    Assumes Y is the up axis (rotation around Y).
    """
    w = q.w
    x = q.x
    y = q.y
    z = q.z
    # Yaw (around Y-axis)
    siny_cosp = 2 * (w * y + x * z)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw_rad = math.atan2(siny_cosp, cosy_cosp)
    return math.degrees(yaw_rad)

def yaw_to_quaternion(yaw_deg):
    """
    Convert a yaw angle in degrees to a quaternion.quaternion,
    rotating around the Y axis.
    """
    yaw_rad = math.radians(yaw_deg)
    half_yaw = yaw_rad / 2
    w = math.cos(half_yaw)
    x = 0
    y = math.sin(half_yaw)
    z = 0
    return quaternion.quaternion(w, x, y, z)


### Vector Processing Functions ###
def normalize_array(arr, eps=1e-5):
    """Normalize a NumPy array to the [0, 1] range, with edge-case handling."""
    arr = np.array(arr)
    min_val, max_val = np.min(arr), np.max(arr)
    if abs(max_val - min_val) > eps:
        return (arr - min_val) / (max_val - min_val)
    else:
        return np.zeros_like(arr)

def process_metrics(distances, avg_entropies, avg_target_probs):
    """Convert to arrays and normalize all three metrics."""
    distances = normalize_array(distances)
    avg_entropies = normalize_array(avg_entropies)
    avg_target_probs = normalize_array(avg_target_probs)
    return distances, avg_entropies, avg_target_probs
