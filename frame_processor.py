# ==============================================================================
# FILE: preprocess.py
# ==============================================================================
import numpy as np
from data_models import Frame
from config import *
from utils import visualize_gradient_in_terminal, find_path_to_nearest_frontier
import collections

## pixel_view 
## 1 地图障碍物 （3*50） * （3*50） pixelgrid
## 4 危险区域 3*3 pixelgrid

def pixels_to_grid(pixel_x: int, pixel_y: int) -> tuple[int, int]:
    """Converts pixel coordinates to grid coordinates."""
    return pixel_x // 50, pixel_y // 50

def create_pixel_view(frame: Frame, grid_view: np.ndarray) -> np.ndarray:
    """
    Creates a high-resolution pixel view by sampling from the pre-calculated grid view.
    This function is optimized using NumPy vectorization to avoid Python loops.
    """
    # 1. Generate coordinate grids for the entire pixel view
    vx_px_arr, vy_px_arr = np.meshgrid(np.arange(PIXEL_VIEW_SIZE), np.arange(PIXEL_VIEW_SIZE))

    # 2. Calculate absolute world pixel coordinates for the entire view
    player_pixel_pos = frame.my_player.position
    view_half_size = PIXEL_VIEW_SIZE // 2
    view_min_x_px = player_pixel_pos.x - view_half_size
    view_min_y_px = player_pixel_pos.y - view_half_size
    
    abs_px_x_arr = view_min_x_px + vx_px_arr
    abs_px_y_arr = view_min_y_px + vy_px_arr

    # 3. Convert world pixel coordinates to world grid coordinates
    world_gx_arr, world_gy_arr = pixels_to_grid(abs_px_x_arr, abs_px_y_arr)

    # 4. Convert world grid coordinates to 11x11 view grid coordinates
    player_grid_x, player_grid_y = pixels_to_grid(player_pixel_pos.x, player_pixel_pos.y)
    view_center_gx, view_center_gy = 5, 5
    
    view_gx_arr = world_gx_arr - player_grid_x + view_center_gx
    view_gy_arr = world_gy_arr - player_grid_y + view_center_gy

    # 5. Clip coordinates to be within the bounds of grid_view [0, 10] to prevent indexing errors.
    # This is a safe implementation detail that doesn't change the logic, as grid_view
    # has already handled out-of-bounds values at its edges.
    view_gx_arr = np.clip(view_gx_arr, 0, VIEW_SIZE - 1)
    view_gy_arr = np.clip(view_gy_arr, 0, VIEW_SIZE - 1)

    # 6. Use advanced indexing to sample from grid_view in a single operation
    pixel_view = np.zeros((PIXEL_CHANNELS, PIXEL_VIEW_SIZE, PIXEL_VIEW_SIZE), dtype=np.float32)
    pixel_view[0] = grid_view[0, view_gy_arr, view_gx_arr]
    pixel_view[1] = grid_view[2, view_gy_arr, view_gx_arr]

    # 7. Add the third channel for the player's own position
    # The player occupies the central 50x50 area of the 100x100 pixel view.
    center_start = PIXEL_VIEW_SIZE // 2 - PIXEL_PER_CELL // 2
    center_end = PIXEL_VIEW_SIZE // 2 + PIXEL_PER_CELL // 2
    pixel_view[2, center_start:center_end, center_start:center_end] = 1.0
                    
    return pixel_view

## grid_view 含义
## 0 地图障碍物 0（P） 可通过 0.5（D）软墙 1.0 （I，N） 硬墙 11*11 grid
## 1 炸弹区域 1.0 has 0.0 none
## 2 危险区域 11 * 11 grid    2/（1 + remaintick） （normalize）

## 3 道具鞋子
## 4 道具药水
## 5 道具炸药包
## 6 加速区域
## 7 减速区域
## 8 我方占领区域
## 9 敌方占领区域c

def create_grid_view(frame: Frame) -> np.ndarray:
    """
    Creates the 11x11 tactical grid view centered on the player.
    """
    current_tick = frame.current_tick
    grid_view = np.zeros((MAP_CHANNELS, VIEW_SIZE, VIEW_SIZE), dtype=np.float32)
    player_grid_x, player_grid_y = pixels_to_grid(frame.my_player.position.x, frame.my_player.position.y)
    view_center_x, view_center_y = 5, 5

    # Channel 0: Terrain
    for view_y in range(VIEW_SIZE):
        for view_x in range(VIEW_SIZE):
            map_x = player_grid_x + (view_x - view_center_x)
            map_y = player_grid_y + (view_y - view_center_y)
            if 0 <= map_y < 16 and 0 <= map_x < 28:
                cell = frame.map[map_y][map_x]
                if cell.terrain == 'D':
                    grid_view[0, view_y, view_x] = 0.5
                elif cell.terrain in ['I', 'N']:
                    grid_view[0, view_y, view_x] = 1.0
            else:
                grid_view[0, view_y, view_x] = 1.0

    # Channel 1: Bombs
    for bomb in frame.bombs:
        view_x = bomb.position.x - player_grid_x + view_center_x
        view_y = bomb.position.y - player_grid_y + view_center_y

        if 0 <= view_x < VIEW_SIZE and 0 <= view_y < VIEW_SIZE:
            grid_view[1, view_y, view_x] = 1.0

    # Channel 2: Danger Zone (with comprehensive chain reaction simulation)
    all_explosion_chains = []
    if frame.bombs:
        bombs_by_pos = { (b.position.x, b.position.y): b for b in frame.bombs }
        unvisited_bombs = set(frame.bombs)
        
        while unvisited_bombs:
            q = [unvisited_bombs.pop()]
            chain = {'bombs': set(q), 'danger_value': 0}
            head = 0
            
            while head < len(q):
                bomb = q[head]; head += 1
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    for i in range(1, bomb.range + 1):
                        nx, ny = bomb.position.x + dx * i, bomb.position.y + dy * i
                        if not (0 <= ny < MAP_HEIGHT and 0 <= nx < MAP_WIDTH): break
                        if (nx, ny) in bombs_by_pos:
                            chained_bomb = bombs_by_pos[(nx, ny)]
                            if chained_bomb in unvisited_bombs:
                                unvisited_bombs.remove(chained_bomb)
                                chain['bombs'].add(chained_bomb)
                                q.append(chained_bomb)
                        if frame.map[ny][nx].terrain in ['I', 'N', 'D']: break
            
            first_tick = min(b.explode_at for b in chain['bombs'])
            # Correctly calculate remaining ticks until explosion
            time_to_explosion = max(0.0, first_tick - current_tick)
            chain['danger_value'] = 1.0 / (1.0 + time_to_explosion)
            all_explosion_chains.append(chain)

            for bomb in chain['bombs']:
                view_x = bomb.position.x - player_grid_x + view_center_x
                view_y = bomb.position.y - player_grid_y + view_center_y
                if 0 <= view_x < VIEW_SIZE and 0 <= view_y < VIEW_SIZE:
                    grid_view[2, view_y, view_x] = max(grid_view[2, view_y, view_x], chain['danger_value'])
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    for i in range(1, bomb.range + 1):
                        nx, ny = bomb.position.x + dx * i, bomb.position.y + dy * i
                        if not (0 <= ny < MAP_HEIGHT and 0 <= nx < MAP_WIDTH): break
                        if frame.map[ny][nx].terrain in ['I', 'N', 'D']: break
                        view_nx = nx - player_grid_x + view_center_x
                        view_ny = ny - player_grid_y + view_center_y
                        if 0 <= view_nx < VIEW_SIZE and 0 <= view_ny < VIEW_SIZE:
                            grid_view[2, view_ny, view_nx] = max(grid_view[2, view_ny, view_nx], chain['danger_value'])
        
        if DEBUG_DANGER_ZONE:
            # Check if there is any danger to print.
            if np.any(grid_view[2] > 0):
                print(f"--- Danger Zone (Tick: {current_tick}) ---")
                # Flip the array vertically to display with (0,0) at the bottom-left.
                flipped_danger_zone = np.flipud(grid_view[2])
                # Use np.round to make the output cleaner.
                print(np.round(flipped_danger_zone, 2))
                print("------------------------------------")

    
    # Channels 3, 4, 5: Items (One-Hot Encoded)
    for item in frame.map_items:
        view_x = item.position.x - player_grid_x + view_center_x
        view_y = item.position.y - player_grid_y + view_center_y
        
        if 0 <= view_x < VIEW_SIZE and 0 <= view_y < VIEW_SIZE:
            if item.type == 'AB':
                grid_view[3, view_y, view_x] = 1.0
            elif item.type == 'SP':
                grid_view[4, view_y, view_x] = 1.0
            elif item.type == 'BP':
                grid_view[5, view_y, view_x] = 1.0

    # Channels 6-9: Special Terrains and Occupied Zones
    for view_y in range(VIEW_SIZE):
        for view_x in range(VIEW_SIZE):
            map_x = player_grid_x + (view_x - view_center_x)
            map_y = player_grid_y + (view_y - view_center_y)
            
            if 0 <= map_y < 16 and 0 <= map_x < 28:
                cell = frame.map[map_y][map_x]
                if cell.terrain in ['I','N','D']: continue
                # Channel 6, 7: Special Terrains (Separated)
                if cell.terrain == 'B':
                    grid_view[6, view_y, view_x] = 1.0
                elif cell.terrain == 'M':
                    grid_view[7, view_y, view_x] = 1.0
                
                # Channel 8, 9: Occupied Zones (Separated)
                if cell.ownership != 'N' and cell.ownership != frame.my_player.team:
                    grid_view[8, view_y, view_x] = 1.0  # Enemy occupied
                elif cell.ownership == 'N':
                    grid_view[9, view_y, view_x] = 1.0  # non occupied

    # Channel 10: Gradient field based on the path to the nearest frontier.
    reconstructed_path, target = find_path_to_nearest_frontier(frame)
    
    full_gradient_map = np.full((MAP_HEIGHT, MAP_WIDTH), -1.0, dtype=np.float32)
    if target:
        path_len = len(reconstructed_path)
        if path_len > 1:
            for i, (r, c) in enumerate(reconstructed_path):
                # The gradient starts at 1.0 at the target and decreases towards the player.
                gradient_value = 1.0 - ((path_len - 1 - i) / (path_len - 1))
                full_gradient_map[r, c] = gradient_value
        elif path_len == 1:
            r, c = reconstructed_path[0]
            full_gradient_map[r, c] = 1.0

    # 4. Sample the 11x11 view from the full gradient map.
    view_y_arr, view_x_arr = np.meshgrid(np.arange(VIEW_SIZE), np.arange(VIEW_SIZE), indexing='ij')
    map_x_arr = player_grid_x + (view_x_arr - view_center_x)
    map_y_arr = player_grid_y + (view_y_arr - view_center_y)

    map_x_arr_clipped = np.clip(map_x_arr, 0, MAP_WIDTH - 1)
    map_y_arr_clipped = np.clip(map_y_arr, 0, MAP_HEIGHT - 1)
    
    sampled_gradients = full_gradient_map[map_y_arr_clipped, map_x_arr_clipped]
    
    # Convert -1 (no gradient) to 0 for the model input
    sampled_gradients[sampled_gradients == -1.0] = 0.0
    
    out_of_bounds_mask = (map_x_arr < 0) | (map_x_arr >= MAP_WIDTH) | \
                         (map_y_arr < 0) | (map_y_arr >= MAP_HEIGHT)
    sampled_gradients[out_of_bounds_mask] = 0.0
    grid_view[10] = sampled_gradients

    # --- DEBUG VISUALIZATION ---
    if DEBUG_VISUALIZE_GRADIENT and frame.my_player.id == 4:
        visualize_gradient_in_terminal(sampled_gradients, frame)

    return grid_view

def create_player_state(frame: Frame) -> np.ndarray:
    """
    Creates the normalized vector of player statistics based on their real effects.
    """
    player = frame.my_player
    
    # 1. Calculate current bomb count based on placed bombs
    current_bomb_count = sum(1 for bomb in frame.bombs if bomb.owner_id == player.id)
    
    # 2. Calculate real bomb range
    # sweet_potion_count starts at 1 and includes the base range.
    bomb_range = player.sweet_potion_count * RANGE_PER_POTION
    
    # 3. Calculate base speed
    speed = BASE_SPEED + player.agility_boots_count * SPEED_PER_BOOT
    
    # 4. Check for speed modifications from terrain
    # We check the four corners of the player's 50x50 grid cell for terrain effects.
    player_pos = player.position
    # A 50x50 box centered on the player will have corners at an offset of roughly +/- 24 pixels.

    corners = [
        (player_pos.x - 25, player_pos.y - 25), (player_pos.x + 24, player_pos.y - 25),
        (player_pos.x - 25, player_pos.y + 24), (player_pos.x + 24, player_pos.y + 24)
    ]
    
    on_acceleration = False
    on_deceleration = False
    for corner_x, corner_y in corners:
        grid_x, grid_y = pixels_to_grid(corner_x, corner_y)
        if 0 <= grid_y < 16 and 0 <= grid_x < 28:
            terrain = frame.map[grid_y][grid_x].terrain
            if terrain == 'B':
                on_acceleration = True
                break
            elif terrain == 'M':
                on_deceleration = True
                break
    
    if on_acceleration:
        speed *= 2.0
    elif on_deceleration:
        speed /= 2.0
        
    # Clip bomb_range before normalization to ensure it doesn't exceed MAX_RANGE,
    # as the actual max possible range is (1 + MAX_POTION) * RANGE_PER_POTION.
    normalized_bomb_range = np.clip(bomb_range, 0, MAX_RANGE) / MAX_RANGE

    return np.array([
        player.bomb_pack_count / MAX_BOMB_PACK,
        normalized_bomb_range,
        speed / MAX_SPEED,
        1.0 if current_bomb_count < player.bomb_pack_count else 0.0,
        1.0 if player.status == 'D' else 0.0
    ], dtype=np.float32)

def preprocess_observation_dict(frame: Frame) -> dict:
    """
    Converts a Frame object into a dictionary of observations for the model by
    calling specialized functions for each component.
    """
    # --- Create each component of the observation ---
    grid_view = create_grid_view(frame)
    pixel_view = create_pixel_view(frame, grid_view)
    player_state = create_player_state(frame)

    # --- Assemble Final Observation ---
    return {
        "grid_view": grid_view,
        "pixel_view": pixel_view,
        "player_state": player_state
    }
