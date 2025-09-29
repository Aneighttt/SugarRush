# ==============================================================================
# FILE: preprocess.py
# ==============================================================================
import numpy as np
from data_models import Frame
from config import *

## pixel_view 
## 1 地图障碍物 （3*50） * （3*50） pixelgrid
## 4 危险区域 3*3 pixelgrid

def pixels_to_grid(pixel_x: int, pixel_y: int) -> tuple[int, int]:
    """Converts pixel coordinates to grid coordinates."""
    return pixel_x // 50, pixel_y // 50

def create_pixel_view(frame: Frame, grid_view: np.ndarray) -> np.ndarray:
    """
    Creates a high-resolution pixel view by sampling from the pre-calculated grid view,
    ensuring consistency between the two representations.
    """
    pixel_view = np.zeros((PIXEL_CHANNELS, PIXEL_VIEW_SIZE, PIXEL_VIEW_SIZE), dtype=np.float32)
    player_pixel_pos = frame.my_player.position
    view_half_size = PIXEL_VIEW_SIZE // 2

    # The top-left corner of the pixel view in absolute world coordinates
    view_min_x_px = player_pixel_pos.x - view_half_size
    view_min_y_px = player_pixel_pos.y - view_half_size
    
    # Player's grid position, used to map world grid to view grid
    player_grid_x, player_grid_y = pixels_to_grid(player_pixel_pos.x, player_pixel_pos.y)
    view_center_gx, view_center_gy = 5, 5

    # Iterate through each pixel of the view
    for vy_px in range(PIXEL_VIEW_SIZE):
        for vx_px in range(PIXEL_VIEW_SIZE):
            # Absolute world coordinate of the current pixel
            abs_px_x = view_min_x_px + vx_px
            abs_px_y = view_min_y_px + vy_px
            
            # Grid coordinate in the world map
            world_gx, world_gy = pixels_to_grid(abs_px_x, abs_px_y)
            
            # Corresponding coordinate in the 11x11 grid_view
            view_gx = world_gx - player_grid_x + view_center_gx
            view_gy = world_gy - player_grid_y + view_center_gy
            
            # Directly sample from the grid_view.
            # The logic assumes that any coordinate that would be out of bounds
            # in the grid_view has already been correctly handled during grid_view's creation.
            pixel_view[0, vy_px, vx_px] = grid_view[0, view_gy, view_gx]
            pixel_view[1, vy_px, vx_px] = grid_view[2, view_gy, view_gx]
                    
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
    grid_view = np.zeros((MAP_CHANNELS, MAP_HEIGHT, MAP_WIDTH), dtype=np.float32)
    player_grid_x, player_grid_y = pixels_to_grid(frame.my_player.position.x, frame.my_player.position.y)
    view_center_x, view_center_y = 5, 5

    # Channel 0: Terrain
    for view_y in range(MAP_HEIGHT):
        for view_x in range(MAP_WIDTH):
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

        if 0 <= view_x < MAP_WIDTH and 0 <= view_y < MAP_HEIGHT:
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
                        if not (0 <= ny < 16 and 0 <= nx < 28): break
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
            chain['danger_value'] = 2.0 / (1.0 + time_to_explosion)
            all_explosion_chains.append(chain)

            for bomb in chain['bombs']:
                view_x = bomb.position.x - player_grid_x + view_center_x
                view_y = bomb.position.y - player_grid_y + view_center_y
                if 0 <= view_x < MAP_WIDTH and 0 <= view_y < MAP_HEIGHT:
                    grid_view[2, view_y, view_x] = max(grid_view[2, view_y, view_x], chain['danger_value'])
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    for i in range(1, bomb.range + 1):
                        nx, ny = bomb.position.x + dx * i, bomb.position.y + dy * i
                        if not (0 <= ny < 16 and 0 <= nx < 28): break
                        if frame.map[ny][nx].terrain in ['I', 'N', 'D']: break
                        view_nx = nx - player_grid_x + view_center_x
                        view_ny = ny - player_grid_y + view_center_y
                        if 0 <= view_nx < MAP_WIDTH and 0 <= view_ny < MAP_HEIGHT:
                            grid_view[2, view_ny, view_nx] = max(grid_view[2, view_ny, view_nx], chain['danger_value'])
                        

    
    # Channels 3, 4, 5: Items (One-Hot Encoded)
    for item in frame.map_items:
        view_x = item.position.x - player_grid_x + view_center_x
        view_y = item.position.y - player_grid_y + view_center_y
        
        if 0 <= view_x < MAP_WIDTH and 0 <= view_y < MAP_HEIGHT:
            if item.type == 'AB':
                grid_view[3, view_y, view_x] = 1.0
            elif item.type == 'SP':
                grid_view[4, view_y, view_x] = 1.0
            elif item.type == 'BP':
                grid_view[5, view_y, view_x] = 1.0

    # Channels 6-9: Special Terrains and Occupied Zones
    for view_y in range(MAP_HEIGHT):
        for view_x in range(MAP_WIDTH):
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
                if cell.ownership != 'N' and  cell.ownership != frame.my_player.team:
                    grid_view[8, view_y, view_x] = 1.0 # Enemy occupied
                elif cell.ownership == 'N':
                    grid_view[9, view_y, view_x] = 1.0 # Self occupied

    # TODO: Implement logic for channels 10-11 here

    return grid_view

def create_player_state(frame: Frame) -> np.ndarray:
    """
    Creates the normalized vector of player statistics based on their real effects.
    """
    player = frame.my_player
    
    # 1. Calculate current bomb count based on placed bombs
    current_bomb_count = sum(1 for bomb in frame.bombs if bomb.owner_id == player.id)
    
    # 2. Calculate real bomb range
    bomb_range = BASE_RANGE + player.sweet_potion_count * RANGE_PER_POTION
    
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
        
    return np.array([
        player.bomb_pack_count / MAX_BOMB_PACK,
        bomb_range / MAX_RANGE,
        speed / MAX_SPEED,
        current_bomb_count / MAX_CURRENT_BOMBS,
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
