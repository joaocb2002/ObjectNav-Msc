# constants.py
# Constants for Object Navigation in a simulated environment

# Robotic agent properties
AGENT_RADIUS = 0.15 # 15 cm
AGENT_HEIGHT = 1.0 # 100 cm

# Sensor observations scale
OBS_SCALE = 9/16 # WIDTH/HEIGHT

# Number of classes
NUM_CLASSES = 27

# Maps sizes
CELL_SIDE = 2*AGENT_RADIUS # 30 cm
MAP_RESOLUTION = 0.01 # 1 cm

# Simulation parameters
CONFIDENCE_THRESHOLD = 0.90
MAX_ITER = float("inf") # No limit on iterations by default

# VALID ACTIONS
ACTIONS = {
    "move_forward",
    "move_backward",
    "move_left",
    "move_right",
    "turn_around",
    "turn_left",
    "turn_right",
}

REDUCED_ACTIONS = {
    "move_forward",
    "turn_left",
    "turn_right",
}

# Dirichlet distribution parameters
DIRICHLET_PRIOR = 1.0

# Display settings
DISPLAY_STEP = 10