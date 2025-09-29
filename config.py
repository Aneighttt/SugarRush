# ==============================================================================
# FILE: config.py
# PURPOSE: Central configuration file for shared constants to avoid circular imports.
# ==============================================================================

# --- Training Parameters ---
TRAIN_BATCH_SIZE = 128

# --- Grid and View Dimensions ---
FRAME_STACK_SIZE = 4
MAP_WIDTH = 11
MAP_HEIGHT = 11
# Updated to match the current implementation (Channels 0-9)
MAP_CHANNELS = 10
PLAYER_STATE_SIZE = 5
PIXEL_PER_CELL = 50
# --- Pixel View Configuration ---
PIXEL_VIEW_SIZE = 100
PIXEL_CHANNELS = 2 # 0: Terrain, 1: Danger Zone

# --- Stacked Channel Calculation ---
# These are derived from the constants above
STACKED_MAP_CHANNELS = MAP_CHANNELS * FRAME_STACK_SIZE
STACKED_PIXEL_CHANNELS = PIXEL_CHANNELS * FRAME_STACK_SIZE

# --- Normalization Constants ---
# These values are used to scale player stats to a [0, 1] range.
MAX_BOMB_PACK = 5.0
MAX_POTION = 5.0
MAX_BOOTS = 5.0
# Max current bombs is 1 (initial) + max upgrades
MAX_CURRENT_BOMBS = 1.0 + MAX_BOMB_PACK
# New constants for real effects
BASE_SPEED = 10.0
SPEED_PER_BOOT = 2.0
MAX_SPEED = (BASE_SPEED + MAX_BOOTS * SPEED_PER_BOOT) * 2.0 # *2 for acceleration point
BASE_RANGE = 1.0
RANGE_PER_POTION = 1.0
MAX_RANGE = BASE_RANGE + MAX_POTION * RANGE_PER_POTION

# --- Action Space Configuration ---
# We use a MultiDiscrete space to decouple movement and bombing decisions.
# [5] for movement: 0=Stop, 1=Up, 2=Down, 3=Left, 4=Right
# [2] for bombing: 0=No, 1=Yes
ACTION_SPACE_DEFINITION = [5, 2]
