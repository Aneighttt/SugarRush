import numpy as np
import data_models

# --- Constants ---
MAP_HEIGHT = 16
MAP_WIDTH = 28
PIXEL_PER_CELL = 50

# --- Action Space Definition ---
# 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT, 4: BOMB, 5: STAY
ACTION_SIZE = 6

def get_occupied_grids(pixel_pos):
    """
    Calculates all grid cells that a 50x50 player body overlaps with,
    given the player's center pixel position.
    """
    center_x, center_y = pixel_pos.x, pixel_pos.y
    
    # Player's body is a 50x50 square centered at (center_x, center_y).
    # The bounding box is from (center_x - 25, center_y - 25) to (center_x + 24, center_y + 24).
    min_x, max_x = center_x - 25, center_x + 24
    min_y, max_y = center_y - 25, center_y + 24

    start_gx = int(min_x / PIXEL_PER_CELL)
    end_gx = int(max_x / PIXEL_PER_CELL)
    start_gy = int(min_y / PIXEL_PER_CELL)
    end_gy = int(max_y / PIXEL_PER_CELL)

    occupied = set()
    for gx in range(start_gx, end_gx + 1):
        for gy in range(start_gy, end_gy + 1):
            # Clamp coordinates to be within map bounds
            if 0 <= gx < MAP_WIDTH and 0 <= gy < MAP_HEIGHT:
                occupied.add((gx, gy))
    return list(occupied)

def get_grid_position(pixel_pos):
    """Converts pixel coordinates of a center point to a single grid coordinate."""
    grid_x = int(pixel_pos.x / PIXEL_PER_CELL)
    grid_y = int(pixel_pos.y / PIXEL_PER_CELL)
    return grid_x, grid_y

def preprocess_frame(frame: data_models.Frame):
    """
    Converts a raw frame object from the server into a state representation
    for the neural network, based on the 2v2 territory control rules.

    Args:
        frame (data_models.Frame): The frame object.

    Returns:
        np.array: A multi-channel numpy array representing the game state.
    """
    # 0: Terrain (0: empty, 0.5: destructible, 1: indestructible)
    # 1: My Player position
    # 2: Teammate positions
    # 3: Enemy positions
    # 4: Bombs (value = normalized time to explosion)
    # 5: Danger zones
    # 6: Items
    # 7: My team's territory
    # 8: Enemy team's territory
    # 9: Special terrain (e.g., slowdown)
    # 10: My player's invincibility status
    num_channels = 11
    state = np.zeros((num_channels, MAP_HEIGHT, MAP_WIDTH), dtype=np.float32)
    my_team_id = frame.my_player.team

    # Channel 0: Terrain, Channel 7/8: Territory, Channel 9: Special Terrain
    for y, row in enumerate(frame.map):
        for x, cell in enumerate(row):
            # Convert server's bottom-left origin to numpy's top-left origin
            ny = MAP_HEIGHT - 1 - y

            # Channel 0: Terrain
            if cell.terrain in ['I', 'N']:  # Indestructible
                state[0, ny, x] = 1.0
            elif cell.terrain == 'D':  # Destructible
                state[0, ny, x] = 0.5
            
            # Channel 9: Special Terrain
            if cell.terrain == 'B':  # Acceleration Point
                state[9, ny, x] = 1.0
            elif cell.terrain == 'M':  # Deceleration Point
                state[9, ny, x] = 0.5

            # Channel 7/8: Territory
            if cell.ownership == my_team_id:
                state[7, ny, x] = 1.0
            elif cell.ownership is not None:  # Belongs to the other team
                state[8, ny, x] = 1.0

    # Channel 1: My Player position (representing all occupied cells)
    for gx, gy in get_occupied_grids(frame.my_player.position):
        ny = MAP_HEIGHT - 1 - gy
        state[1, ny, gx] = 1.0

    # Channel 2: Teammates & Channel 3: Enemies (representing all occupied cells)
    for other_player in frame.other_players:
        for gx, gy in get_occupied_grids(other_player.position):
            ny = MAP_HEIGHT - 1 - gy
            if other_player.team == my_team_id:
                state[2, ny, gx] = 1.0  # Teammate
            else:
                state[3, ny, gx] = 1.0  # Enemy

    # Channels 4 & 5: Bombs and Danger Zones
    for bomb in frame.bombs:
        # FIX: bomb.position is already in grid coordinates, no conversion needed.
        bomb_x, bomb_y = bomb.position.x, bomb.position.y
        ny_bomb = MAP_HEIGHT - 1 - bomb_y
        time_to_explode = bomb.explode_at - frame.current_tick
        state[4, ny_bomb, bomb_x] = max(0, 1.0 - (time_to_explode / 100.0))

        state[5, ny_bomb, bomb_x] = 1.0
        for i in range(1, bomb.range + 1): # Right
            if bomb_x + i >= MAP_WIDTH or state[0, ny_bomb, bomb_x + i] == 1.0: break
            state[5, ny_bomb, bomb_x + i] = 1.0
        for i in range(1, bomb.range + 1): # Left
            if bomb_x - i < 0 or state[0, ny_bomb, bomb_x - i] == 1.0: break
            state[5, ny_bomb, bomb_x - i] = 1.0
        for i in range(1, bomb.range + 1): # Up (in numpy: y decreases)
            if ny_bomb - i < 0 or state[0, ny_bomb - i, bomb_x] == 1.0: break
            state[5, ny_bomb - i, bomb_x] = 1.0
        for i in range(1, bomb.range + 1): # Down (in numpy: y increases)
            if ny_bomb + i >= MAP_HEIGHT or state[0, ny_bomb + i, bomb_x] == 1.0: break
            state[5, ny_bomb + i, bomb_x] = 1.0

    # Channel 6: Items
    for item in frame.map_items:
        # FIX: Assuming item.position is also in grid coordinates.
        item_x, item_y = item.position.x, item.position.y
        ny_item = MAP_HEIGHT - 1 - item_y
        state[6, ny_item, item_x] = 1.0
        
    # Channel 10: My player's invincibility status
    # If invincible, fill the entire channel with 1.0 to give a strong signal to the CNN
    if 'INVINCIBLE' in [s.name for s in frame.my_player.extra_status]:
        state[10, :, :] = 1.0

    return state
