from data_models import Frame
from config import *
import collections
from utils import calculate_distance_map_to_frontier

def count_territory(frame: Frame):
    my_territory = 0
    enemy_territory = 0
    my_team_id = frame.my_player.team
    
    enemy_team_id = next((player.team for player in frame.other_players if player.team != my_team_id), 'N')
    for row in frame.map:
        for cell in row:
            if cell.ownership == my_team_id:
                my_territory += 1
            elif cell.ownership == enemy_team_id:
                enemy_territory += 1
    return my_territory, enemy_territory

def get_occupied_grids_from_position(pixel_pos):
    center_x, center_y = pixel_pos.x, pixel_pos.y
    min_x, max_x = center_x - 25, center_x + 24
    min_y, max_y = center_y - 25, center_y + 24

    start_gx = int(min_x / PIXEL_PER_CELL)
    end_gx = int(max_x / PIXEL_PER_CELL)
    start_gy = int(min_y / PIXEL_PER_CELL)
    end_gy = int(max_y / PIXEL_PER_CELL)

    occupied = set()
    for gx in range(start_gx, end_gx + 1):
        for gy in range(start_gy, end_gy + 1):
            if 0 <= gx < MAP_WIDTH and 0 <= gy < MAP_HEIGHT:
                occupied.add((gx, gy))
    return list(occupied)

def _get_explosion_grids(bomb, frame: Frame):
    explosion_grids = set()
    bomb_x, bomb_y = bomb.position.x, bomb.position.y
    
    explosion_grids.add((bomb_x, bomb_y))

    for i in range(1, bomb.range + 1):
        x = bomb_x + i
        if not (0 <= x < MAP_WIDTH): break
        explosion_grids.add((x, bomb_y))
        if frame.map[bomb_y][x].terrain in ['I', 'N', 'D']: break
    for i in range(1, bomb.range + 1):
        x = bomb_x - i
        if not (0 <= x < MAP_WIDTH): break
        explosion_grids.add((x, bomb_y))
        if frame.map[bomb_y][x].terrain in ['I', 'N', 'D']: break
    for i in range(1, bomb.range + 1):
        y = bomb_y + i
        if not (0 <= y < MAP_HEIGHT): break
        explosion_grids.add((bomb_x, y))
        if frame.map[y][bomb_x].terrain in ['I', 'N', 'D']: break
    for i in range(1, bomb.range + 1):
        y = bomb_y - i
        if not (0 <= y < MAP_HEIGHT): break
        explosion_grids.add((bomb_x, y))
        if frame.map[y][bomb_x].terrain in ['I', 'N', 'D']: break
    #print(bomb_x,bomb_y,explosion_grids)  
    return explosion_grids

def get_danger_zones(frame: Frame):
    danger_zones = set()
    for bomb in frame.bombs:
        danger_zones.update(_get_explosion_grids(bomb, frame))
    return danger_zones


def _find_nearest_safe_tile_coords(start_gx, start_gy, danger_zones, max_dist=100):
    """
    Finds the coordinates of the nearest tile not in the danger_zones using BFS.
    """
    q = collections.deque([(start_gx, start_gy, 0)])
    visited = set([(start_gx, start_gy)])

    if (start_gx, start_gy) not in danger_zones:
        return (start_gx, start_gy)

    while q:
        x, y, dist = q.popleft()

        if dist >= max_dist:
            continue

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy

            if 0 <= nx < MAP_WIDTH and 0 <= ny < MAP_HEIGHT and (nx, ny) not in visited:
                if (nx, ny) not in danger_zones:
                    return (nx, ny)
                visited.add((nx, ny))
                q.append((nx, ny, dist + 1))
    
    return None


def _find_nearest_unclaimed_tile_coords(frame: Frame, start_gx: int, start_gy: int):
    """
    Finds the coordinates of the nearest tile not owned by the player's team using BFS.
    Returns None if all tiles are owned.
    """
    my_team_id = frame.my_player.team
    q = [(start_gx, start_gy)]
    visited = set(q)

    head = 0
    while head < len(q):
        x, y = q[head]
        head += 1

        if frame.map[y][x].ownership != my_team_id:
            return (x, y)

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < MAP_WIDTH and 0 <= ny < MAP_HEIGHT and (nx, ny) not in visited:
                visited.add((nx, ny))
                q.append((nx, ny))
    
    return None

3
def calculate_reward(previous_frame: Frame, current_frame: Frame, previous_action: int):
    rewards = {}

    # --- 1. Territory difference ---
    current_my_territory, current_enemy_territory = count_territory(current_frame)
    prev_my_territory, prev_enemy_territory = count_territory(previous_frame)
    prev_diff = prev_my_territory - prev_enemy_territory
    current_diff = current_my_territory - current_enemy_territory
    territory_reward = (current_diff - prev_diff) * 0.3  # 缩小到 ±0.5
    if territory_reward != 0:
        rewards['territory_diff'] = territory_reward

    # --- 2. Reward for capturing the tile the player is on ---
    my_team_id = current_frame.my_player.team
    curr_pos = current_frame.my_player.position
    curr_gx = curr_pos.x // PIXEL_PER_CELL
    curr_gy = curr_pos.y // PIXEL_PER_CELL
    if 0 <= curr_gx < MAP_WIDTH and 0 <= curr_gy < MAP_HEIGHT:
        prev_own = previous_frame.map[curr_gy][curr_gx].ownership
        curr_own = current_frame.map[curr_gy][curr_gx].ownership
        #print(prev_own, curr_own,curr_gx, curr_gy)
        if prev_own != my_team_id and curr_own == my_team_id:
            rewards['capture_tile'] = 0.6  # Clipped to [-1, 1] range

    # --- 3. End of game reward ---
    if current_frame.current_tick == 1800:
        if current_my_territory > current_enemy_territory:
            rewards['win'] = 1.0  # Clipped to [-1, 1] range
        elif current_my_territory < current_enemy_territory:
            rewards['lose'] = -1.0  # Clipped to [-1, 1] range

    # --- 4. Stun penalty ---
    if current_frame.my_player.status == 'D' and previous_frame.my_player.status == 'A':
        rewards['stun'] = -1.0  # Clipped to [-1, 1] range
    #print(current_frame.my_player.status)
    # --- 5. Item collection reward ---
    curr_items = (current_frame.my_player.bomb_pack_count +
                  current_frame.my_player.sweet_potion_count +
                  current_frame.my_player.agility_boots_count)
    prev_items = (previous_frame.my_player.bomb_pack_count +
                  previous_frame.my_player.sweet_potion_count +
                  previous_frame.my_player.agility_boots_count)
    if curr_items > prev_items:
        rewards['item_collect'] = 0.4  # Clipped to [-1, 1] range

    # --- 6. Bomb placement evaluation (Refactored based on user feedback) ---
    if previous_action == 4:  # Player intended to place a bomb
        player = previous_frame.my_player
        
        # Check if the player was actually able to place a bomb
        my_bombs_on_map = sum(1 for bomb in previous_frame.bombs if bomb.owner_id == player.id)
        
        if my_bombs_on_map < player.bomb_pack_count: # +1 for the base bomb
            strategic_value = 0.0
            has_valid_target = False
            
            # Create a temporary bomb object based on player's state in the previous frame
            bomb_pos_x = player.position.x // PIXEL_PER_CELL
            bomb_pos_y = player.position.y // PIXEL_PER_CELL
            # sweet_potion_count starts at 1 and includes the base range.
            bomb_range = player.sweet_potion_count * RANGE_PER_POTION
            
            # A mock bomb object for calculation. We don't need all attributes.
            temp_bomb = collections.namedtuple('Bomb', ['position', 'range'])
            temp_bomb.position = collections.namedtuple('Position', ['x', 'y'])(bomb_pos_x, bomb_pos_y)
            temp_bomb.range = int(bomb_range)

            # CRITICAL: Explosion area and evaluation must be based on the state when the bomb was placed (previous_frame)
            explosion_area = _get_explosion_grids(temp_bomb, previous_frame)

            # Get enemy positions from the previous frame
            enemy_grids = set()
            for p in previous_frame.other_players:
                if p.team != player.team:
                    enemy_grids.update(get_occupied_grids_from_position(p.position))

            # Evaluate the bomb's strategic value
            for x, y in explosion_area:
                if previous_frame.map[y][x].terrain == 'D':
                    strategic_value += 0.16  # Clipped to [-1, 1] range
                    has_valid_target = True
                if (x, y) in enemy_grids:
                    strategic_value += 0.4  # Clipped to [-1, 1] range
                    has_valid_target = True
            
            # Penalize useless bombs
            if not has_valid_target:
                strategic_value -= 0.3 # Clipped to [-1, 1] range

            # Reward for placing a bomb and having a safe escape route
            player_grids_after_move = set(get_occupied_grids_from_position(current_frame.my_player.position))
            if not player_grids_after_move.isdisjoint(explosion_area): # If player is still in danger
                safe_tile = _find_nearest_safe_tile_coords(bomb_pos_x, bomb_pos_y, explosion_area)
                if safe_tile:
                    strategic_value += 0.1 # Clipped to [-1, 1] range
            
            if strategic_value != 0:
                rewards['bomb_strategy'] = strategic_value
        else:
            # Penalize trying to place a bomb when at max capacity
            rewards['bomb_limit_penalty'] = -1.0

    # --- 7. Reward for moving / Penalty for not moving ---
    is_move = previous_action in [0, 1, 2, 3]
    collided_with_wall = False
    if is_move:
        if current_frame.my_player.position != previous_frame.my_player.position:
            rewards['move_reward'] = 0.02  # Clipped to [-1, 1] range
        else:
            rewards['not_moving'] = -0.7 # Clipped to [-1, 1] range
            collided_with_wall = True
    # --- 8. Penalty for doing nothing ---i bu
    if previous_action == 5:
        rewards['do_nothing'] = -0.2

    # --- 9. Living penalty ---
    rewards['living_penalty'] = -0.001

    # --- 10. Reward for Moving Towards Frontier (REMOVED) ---
    # This reward is now redundant because the agent's observation space
    # includes a pathfinding gradient. The agent can learn to follow this
    # gradient to reach the frontier, making this explicit reward unnecessary.
    # if previous_action in [0, 1, 2, 3] and not collided_with_wall:
    #     ... (code removed) ...

    # --- 11. Reward/Penalty for Entering/Exiting Danger Zone ---
    danger_zones = get_danger_zones(previous_frame)
    
    prev_player_grids = set(get_occupied_grids_from_position(previous_frame.my_player.position))
    curr_player_grids = set(get_occupied_grids_from_position(current_frame.my_player.position))
    
    was_in_danger = not prev_player_grids.isdisjoint(danger_zones)
    is_in_danger = not curr_player_grids.isdisjoint(danger_zones)

    if not was_in_danger and is_in_danger:
        rewards['enter_danger_zone'] = -0.6  # Clipped to [-1, 1] range
    elif was_in_danger and not is_in_danger:
        rewards['exit_danger_zone'] = 0.8   # Clipped to [-1, 1] range

    # --- 12. Continuous Penalty for Staying in Danger Zone ---
    if is_in_danger:
        rewards['staying_in_danger'] = -0.7 # Clipped to [-1, 1] range, must be > capture_tile

        # --- 13. Reward for Moving Towards Safety (when in danger) ---
        if previous_action in [0, 1, 2, 3]: # Only if the agent tried to move
            prev_pos = previous_frame.my_player.position
            prev_gx, prev_gy = prev_pos.x // PIXEL_PER_CELL, prev_pos.y // PIXEL_PER_CELL
            
            curr_pos = current_frame.my_player.position
            curr_gx, curr_gy = curr_pos.x // PIXEL_PER_CELL, curr_pos.y // PIXEL_PER_CELL

            # Find the nearest safe tile from the previous position
            target_coords = _find_nearest_safe_tile_coords(prev_gx, prev_gy, danger_zones)
            
            if target_coords:
                target_sx, target_sy = target_coords
                dist_before = abs(prev_gx - target_sx) + abs(prev_gy - target_sy)
                dist_after = abs(curr_gx - target_sx) + abs(curr_gy - target_sy)

                if dist_after < dist_before:
                    rewards['move_towards_safety'] = 0.8 # Clipped to [-1, 1] range, must be > staying_in_danger

    # --- 14. Reward for Moving in the Direction of the Gradient ---
    # This rewards the agent for moving in the general direction of the gradient,
    # based on pixel-level movement vectors.
    if previous_action in [0, 1, 2, 3] and not collided_with_wall and hasattr(previous_frame, 'distance_map'):
        dist_map = previous_frame.distance_map
        prev_pos = previous_frame.my_player.position
        prev_gx, prev_gy = prev_pos.x // PIXEL_PER_CELL, prev_pos.y // PIXEL_PER_CELL

        # 1. Find the best neighboring grid cell (lowest distance on the map)
        best_neighbor_dist = dist_map[prev_gy, prev_gx]
        best_neighbor_coords = None
        if best_neighbor_dist == -1: # Player is not on a valid path
            return rewards

        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]: # U, D, L, R
            nx, ny = prev_gx + dx, prev_gy + dy
            if 0 <= nx < MAP_WIDTH and 0 <= ny < MAP_HEIGHT:
                neighbor_dist = dist_map[ny, nx]
                if neighbor_dist != -1 and neighbor_dist < best_neighbor_dist:
                    best_neighbor_dist = neighbor_dist
                    best_neighbor_coords = (nx, ny)

        # 2. If a better neighbor exists, compare movement direction
        if best_neighbor_coords:
            # Ideal vector: from player's start pos to center of best neighbor cell
            best_nx, best_ny = best_neighbor_coords
            ideal_target_px_x = best_nx * PIXEL_PER_CELL + PIXEL_PER_CELL / 2
            ideal_target_px_y = best_ny * PIXEL_PER_CELL + PIXEL_PER_CELL / 2
            ideal_vec = (ideal_target_px_x - prev_pos.x, ideal_target_px_y - prev_pos.y)

            # Actual movement vector
            curr_pos = current_frame.my_player.position
            actual_vec = (curr_pos.x - prev_pos.x, curr_pos.y - prev_pos.y)

            # 3. Use dot product to check if vectors are in the same general direction
            dot_product = ideal_vec[0] * actual_vec[0] + ideal_vec[1] * actual_vec[1]

            if dot_product > 0:
                rewards['follow_gradient_direction'] = 0.08

    return rewards
