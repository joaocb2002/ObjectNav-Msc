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
CONFIDENCE_THRESHOLD = 0.75
LOCATION_ERROR_THRESHOLD = 0.50 # 50 cm
MAX_ITER_COEF = 0.75 # Coefficient to determine maximum iterations based on free cells
PSEUDO_COUNT_THRESHOLD = 8.0 # Equivalent to probability of 0.5
NUM_EPOCHS = 100

# VALID ACTIONS
EXTENDED_ACTIONS = {
    "move_forward",
    "move_backward",
    "move_left",
    "move_right",
    "turn_around",
    "turn_left",
    "turn_right",
}

ACTIONS = {
    "move_forward",
    "turn_left",
    "turn_right",
}

# Dirichlet distribution parameters
DIRICHLET_PRIOR = 1.0

# Display settings
DISPLAY_STEP = 10 # Display every 10 actions by default