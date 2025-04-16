import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from io import BytesIO

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
    cv2.putText(image_copy, 'Z', (int(x_end[0] + arrow_len / 3), int(x_end[1] + arrow_len / 5)), font, font_scale, (17, 17, 132), thickness)
    cv2.putText(image_copy, 'X', (int(z_end[0] + arrow_len / 5), int(z_end[1] + arrow_len / 3)), font, font_scale, (17, 17, 132), thickness)

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
