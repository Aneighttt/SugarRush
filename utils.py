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

def _get_nearby_wall_pixels(player_grid_pos, frame: data_models.Frame):
    """
    Gets a set of all pixel coordinates for all wall blocks in a 3x3 area
    around the player, using the correct coordinate system.
    """
    wall_pixels = set()
    px, py = player_grid_pos
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            gx, gy = px + dx, py + dy
            if 0 <= gx < MAP_WIDTH and 0 <= gy < MAP_HEIGHT:
                if frame.map[gy][gx].terrain in ['I', 'N', 'D']:
                    # This is a wall. Its bottom-left pixel is at (gx*50, gy*50).
                    for i in range(PIXEL_PER_CELL):
                        for j in range(PIXEL_PER_CELL):
                            wall_pixels.add((gx * PIXEL_PER_CELL + i, gy * PIXEL_PER_CELL + j))
    return wall_pixels

def _check_pixel_collision(player_pixel_pos, direction, wall_pixel_set):
    """
    Checks if the leading edge of the player's bounding box will collide
    with any pixel in the wall_pixel_set.
    """
    min_x, max_x = player_pixel_pos.x - 25, player_pixel_pos.x + 24
    min_y, max_y = player_pixel_pos.y - 25, player_pixel_pos.y + 24

    if direction == 'U':
        # Check the top edge
        for x in range(min_x, max_x + 1):
            if (x, max_y + 1) in wall_pixel_set: return True
    elif direction == 'D':
        # Check the bottom edge
        for x in range(min_x, max_x + 1):
            if (x, min_y - 1) in wall_pixel_set: return True
    elif direction == 'L':
        # Check the left edge
        for y in range(min_y, max_y + 1):
            if (min_x - 1, y) in wall_pixel_set: return True
    elif direction == 'R':
        # Check the right edge
        for y in range(min_y, max_y + 1):
            if (max_x + 1, y) in wall_pixel_set: return True
            
    return False

def get_grid_position(pixel_pos):
    """Converts pixel coordinates of a center point to a single grid coordinate."""
    grid_x = int(pixel_pos.x / PIXEL_PER_CELL)
    grid_y = int(pixel_pos.y / PIXEL_PER_CELL)
    return grid_x, grid_y

def _get_explosion_grids(bomb, frame: data_models.Frame):
    """(Util) Calculates the grid cells affected by a single bomb's explosion."""
    explosion_grids = set()
    bomb_x, bomb_y = bomb.position.x, bomb.position.y
    
    # --- CRITICAL: The bomb's own location is part of the danger zone ---
    explosion_grids.add((bomb_x, bomb_y))

    # Right
    for i in range(1, bomb.range + 1):
        x = bomb_x + i
        if not (0 <= x < MAP_WIDTH): break
        explosion_grids.add((x, bomb_y))
        if frame.map[bomb_y][x].terrain in ['I', 'N', 'D']: break
    # Left
    for i in range(1, bomb.range + 1):
        x = bomb_x - i
        if not (0 <= x < MAP_WIDTH): break
        explosion_grids.add((x, bomb_y))
        if frame.map[bomb_y][x].terrain in ['I', 'N', 'D']: break
    # Up
    for i in range(1, bomb.range + 1):
        y = bomb_y + i
        if not (0 <= y < MAP_HEIGHT): break
        explosion_grids.add((bomb_x, y))
        if frame.map[y][bomb_x].terrain in ['I', 'N', 'D']: break
    # Down
    for i in range(1, bomb.range + 1):
        y = bomb_y - i
        if not (0 <= y < MAP_HEIGHT): break
        explosion_grids.add((bomb_x, y))
        if frame.map[y][bomb_x].terrain in ['I', 'N', 'D']: break
    return explosion_grids

def _get_full_danger_zones(frame: data_models.Frame):
    """(Util) Calculates all danger zones on the map from all bombs."""
    danger_zones = set()
    for bomb in frame.bombs:
        danger_zones.update(_get_explosion_grids(bomb, frame))
    return danger_zones

def preprocess_frame(frame: data_models.Frame, view_size=11):
    """
    Converts a raw frame into a self-centered, localized state representation.

    Args:
        frame (data_models.Frame): The frame object.
        view_size (int): The width and height of the local view (e.g., 11x11).

    Returns:
        np.array: A multi-channel numpy array representing the local game state.
    """
    num_channels = 11
    # The state is now the size of the local view
    state = np.zeros((num_channels, view_size, view_size), dtype=np.float32)
    my_team_id = frame.my_player.team
    
    # Get my player's grid position, which will be the center of our view
    my_gx, my_gy = get_grid_position(frame.my_player.position)
    
    half_view = view_size // 2


    # Pre-calculate all danger zones on the absolute map
    danger_zones = _get_full_danger_zones(frame)

    # --- Populate channels based on the cell at (map_x, map_y) ---
    # (This loop is now aware of the pre-calculated danger zones)
    for dy in range(-half_view, half_view + 1):
        for dx in range(-half_view, half_view + 1):
            map_x, map_y = my_gx + dx, my_gy + dy
            state_x, state_y = dx + half_view, dy + half_view

            if not (0 <= map_x < MAP_WIDTH and 0 <= map_y < MAP_HEIGHT):
                state[0, state_y, state_x] = 1.0
                continue

            cell = frame.map[map_y][map_x]
            if cell.terrain in ['I', 'N']: state[0, state_y, state_x] = 1.0
            elif cell.terrain == 'D': state[0, state_y, state_x] = 0.5
            if cell.terrain == 'B': state[9, state_y, state_x] = 1.0
            elif cell.terrain == 'M': state[9, state_y, state_x] = 0.5
            # Channel 7: Unified Territory Value Map
            # 0.3 = My Team (low value, but indicates ownership)
            # 0.6 = Neutral (medium value, 1 point swing)
            # 1.0 = Enemy (high value, 2 point swing)
            if cell.ownership == my_team_id:
                state[7, state_y, state_x] = 0.3
            elif cell.ownership == 'N':
                state[7, state_y, state_x] = 0.6
            elif cell.ownership is not None: # Enemy territory
                state[7, state_y, state_x] = 1.0
            
            # Channel 8 is now free and unused.
            
            # Channel 5: Danger Zones (now using the pre-calculated set)
            if (map_x, map_y) in danger_zones:
                state[5, state_y, state_x] = 1.0

    # --- Populate object-based channels relative to our position ---
    
    # Channel 1: My Player's full body position
    my_occupied_grids = get_occupied_grids(frame.my_player.position)
    for gx, gy in my_occupied_grids:
        rel_dx, rel_dy = gx - my_gx, gy - my_gy
        if abs(rel_dx) <= half_view and abs(rel_dy) <= half_view:
            state_x, state_y = rel_dx + half_view, rel_dy + half_view
            state[1, state_y, state_x] = 1.0

    for p in frame.other_players:
        gx, gy = get_grid_position(p.position)
        rel_dx, rel_dy = gx - my_gx, gy - my_gy
        if abs(rel_dx) <= half_view and abs(rel_dy) <= half_view:
            state_x, state_y = rel_dx + half_view, rel_dy + half_view
            if p.team == my_team_id: state[2, state_y, state_x] = 1.0
            else: state[3, state_y, state_x] = 1.0

    # Channel 4: Bombs
    for bomb in frame.bombs:
        gx, gy = bomb.position.x, bomb.position.y
        rel_dx, rel_dy = gx - my_gx, gy - my_gy
        if abs(rel_dx) <= half_view and abs(rel_dy) <= half_view:
            state_x, state_y = rel_dx + half_view, rel_dy + half_view
            time_to_explode = bomb.explode_at - frame.current_tick
            state[4, state_y, state_x] = max(0, 1.0 - (time_to_explode / 100.0))

    # Channel 6: Items
    for item in frame.map_items:
        gx, gy = item.position.x, item.position.y
        rel_dx, rel_dy = gx - my_gx, gy - my_gy
        if abs(rel_dx) <= half_view and abs(rel_dy) <= half_view:
            state_x, state_y = rel_dx + half_view, rel_dy + half_view
            state[6, state_y, state_x] = 1.0
            
    # Channel 10: My player's invincibility status
    if 'INV' in [s.name for s in frame.my_player.extra_status]:
        state[10, :, :] = 1.0

    # This is a placeholder for where the collision vector would be calculated.
    # The actual calculation and injection happens in ai_logic.py
    # to ensure we have access to the full frame object.

    return state
