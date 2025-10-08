# ==============================================================================
# FILE: config.py
# PURPOSE: Central configuration file for shared constants to avoid circular imports.
# ==============================================================================

# --- Debugging Flags ---
# Set to True to print the pathfinding gradient map to the console for each frame.
# This will significantly slow down the process and spam the console.
DEBUG_VISUALIZE_GRADIENT = False

# Set to True to print the calculated danger zone map (grid_view channel 2) to the console.
DEBUG_DANGER_ZONE = False

# --- Grid and View Dimensions ---
MAP_WIDTH = 28
MAP_HEIGHT = 16
VIEW_SIZE = 11
# Updated to match the current implementation (Channels 0-10)
MAP_CHANNELS = 11
PLAYER_STATE_SIZE = 5
PIXEL_PER_CELL = 50
# --- Pixel View Configuration ---
PIXEL_VIEW_SIZE = 100
PIXEL_CHANNELS = 3 # 0: Terrain, 1: Danger Zone, 2: Self Position

# --- Normalization Constants ---
# These values are used to scale player stats to a [0, 1] range.
MAX_BOMB_PACK = 5.0
MAX_POTION = 5.0
MAX_BOOTS = 5.0
# Max current bombs is 1 (initial) + max upgrades
MAX_CURRENT_BOMBS = MAX_BOMB_PACK
# New constants for real effects
BASE_SPEED = 10.0
SPEED_PER_BOOT = 2.0
MAX_SPEED = (BASE_SPEED + MAX_BOOTS * SPEED_PER_BOOT) * 2.0 # *2 for acceleration point
RANGE_PER_POTION = 1.0
MAX_RANGE = MAX_POTION * RANGE_PER_POTION

# --- Action Space Configuration ---
# The action space is defined in `environment.py` as `gym.spaces.Discrete(6)`.
# The mapping from action index to game command is handled in `robot.py`.
# 0: Up
# 1: Down
# 2: Left
# 3: Right
# 4: Place Bomb
# 5: Stay (No-op)
