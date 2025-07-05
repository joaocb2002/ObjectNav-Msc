import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import quaternion
from scipy.ndimage import label, binary_dilation

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

    # Define text offsets

    # Draw labels
    cv2.putText(image_copy, 'X', (int(x_end[0] + arrow_len / 3), int(x_end[1] + arrow_len / 5)), font, font_scale, (17, 17, 132), thickness)
    cv2.putText(image_copy, 'Z', (int(z_end[0] + arrow_len / 5), int(z_end[1] + arrow_len / 3)), font, font_scale, (17, 17, 132), thickness)

    return image_copy

def plot_two_maps(topdown_map: np.ndarray, occ_grid_map: np.ndarray):
    """
    Plots two images side by side: the scene map and the occupancy grid map.

    Args:
        topdown_map (np.ndarray): Image of the full scene or environment.
        occ_grid_map (np.ndarray): Raw occupancy grid (e.g., binary values).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    # Top-down map
    ax1.imshow(topdown_map)
    ax1.set_title('scene map')
    ax1.axis('off')

    # Occupancy grid map
    rows, cols = occ_grid_map.shape[:2]
    ax2.imshow(occ_grid_map, cmap='gray')
    ax2.set_title('occ grid map')
    ax2.axis('off')
    for row in range(1, rows):
        ax2.axhline(y=row-0.5, color='black', linewidth=0.5)  # Horizontal grid lines
    for col in range(1, cols):
        ax2.axvline(x=col-0.5, color='black', linewidth=0.5)  # Vertical grid lines

    # Plot red dots at the center of each cell
    for row in range(0, rows):
        for col in range(0, cols):
            # Only plot a red dot for non-zero cells
            if occ_grid_map[row, col, 0] > 128:
                plt.plot(col, row, marker='o', color='red', markersize=1)

    plt.tight_layout()
    plt.show()

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

def bin_index(scale, bin_vector):
    """
    Computes the bin index for a given scale and bin vector.
    """
    bin_index = -1
    for i in range(len(bin_vector)):
        if scale <= bin_vector[i] or scale > bin_vector[-1]:
            break
        bin_index += 1

    if bin_index == -1: bin_index = 0

    return bin_index

import numpy as np
from scipy.ndimage import label, binary_dilation

def retain_largest_white_chunk(grid):
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