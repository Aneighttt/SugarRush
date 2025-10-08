from data_models import Frame
from config import *
import collections
from utils import calculate_distance_map_to_frontier, find_path_to_nearest_frontier
import numpy as np
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



def _calculate_safety_distance_map(grid_view):
    """
    Calculates a distance map where each cell's value is the shortest distance
    to the nearest safe zone using a multi-source BFS.
    """
    danger_map = grid_view[2, :, :]
    terrain_map = grid_view[0, :, :]
    
    # Initialize distances with a large value
    distance_map = np.full((VIEW_SIZE, VIEW_SIZE), 99, dtype=np.int32)
    q = collections.deque()

    # Find all safe zones (sources for the BFS)
    for y in range(VIEW_SIZE):
        for x in range(VIEW_SIZE):
            if danger_map[y, x] == 0 and terrain_map[y, x] != 1.0:
                distance_map[y, x] = 0
                q.append((x, y))

    # Perform multi-source BFS
    while q:
        x, y = q.popleft()
        dist = distance_map[y, x]

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy

            if 0 <= ny < VIEW_SIZE and 0 <= nx < VIEW_SIZE:
                # If the neighbor is reachable and has not been visited yet
                if terrain_map[ny, nx] != 1.0 and distance_map[ny, nx] == 99:
                    distance_map[ny, nx] = dist + 1
                    q.append((nx, ny))
    
    return distance_map


def _get_explosion_grids(bomb, frame):
    """Helper to calculate explosion grids for a bomb."""
    grids = set()
    # Add the bomb's own grid
    grids.add((bomb.position.x, bomb.position.y))
    # Calculate explosion in 4 directions
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        for i in range(1, bomb.range + 1):
            nx, ny = bomb.position.x + dx * i, bomb.position.y + dy * i
            if not (0 <= ny < MAP_HEIGHT and 0 <= nx < MAP_WIDTH):
                break
            grids.add((nx, ny))
            # Stop explosion path if it hits a wall
            if frame.map[ny][nx].terrain in ['I', 'N', 'D']:
                break
    return grids


def calculate_reward(previous_frame: Frame, current_frame: Frame, previous_action: int, previous_processed_obs: dict):
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
            
            # Get enemy positions from the previous frame
            enemy_grids = set()
            for p in previous_frame.other_players:
                if p.team != player.team:
                    enemy_grids.update(get_occupied_grids_from_position(p.position))

            # --- Final, Strict Strategic Bombing Reward ---
            is_strategic = False

            # Rule 1: Check for destructible walls immediately adjacent to the PLAYER
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = bomb_pos_x + dx, bomb_pos_y + dy
                if (0 <= ny < MAP_HEIGHT and 0 <= nx < MAP_WIDTH and
                        previous_frame.map[ny][nx].terrain == 'D'):
                    is_strategic = True
                    break
            
            # Rule 2: Check for any enemy "nearby" (e.g., within a 5x5 box)
            if not is_strategic:
                nearby_range = 2
                for p in previous_frame.other_players:
                    if p.team != player.team:
                        enemy_gx = p.position.x // PIXEL_PER_CELL
                        enemy_gy = p.position.y // PIXEL_PER_CELL
                        if (abs(bomb_pos_x - enemy_gx) <= nearby_range and
                                abs(bomb_pos_y - enemy_gy) <= nearby_range):
                            is_strategic = True
                            break
            
            if is_strategic:
                rewards['bomb_strategy'] = 0.5
            else:
                # This penalty must be extremely severe to make "reward farming" by
                # placing useless bombs and then dodging them an unprofitable strategy.
                # It must outweigh the sum of all potential future rewards from dodging.
                rewards['bomb_strategy'] = -2.0
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
            # A moderate penalty for bumping into walls.
            rewards['not_moving'] = -0.3 # Clipped to [-1, 1] range
            collided_with_wall = True
    # --- 8. Penalty for doing nothing ---i bu
    if previous_action == 5:
        # A gentle nudge to prevent the agent from being passive.
        rewards['do_nothing'] = -0.1

    # --- 9. Living penalty ---
    rewards['living_penalty'] = -0.001

    # --- 10. Reward for Moving Towards Frontier (REMOVED) ---
    # This reward is now redundant because the agent's observation space
    # includes a pathfinding gradient. The agent can learn to follow this
    # gradient to reach the frontier, making this explicit reward unnecessary.
    # if previous_action in [0, 1, 2, 3] and not collided_with_wall:
    #     ... (code removed) ...

    # --- 11. Reward/Penalty for Entering/Exiting Danger Zone ---
    # Use the pre-calculated danger map from the agent's observation
    # The passed `previous_processed_obs` is the stacked observation.
    # We need the most recent half, which is the second part.
    prev_grid_view = previous_processed_obs['grid_view']
    danger_map = prev_grid_view[11 + 2, :, :] # Channel 2 of the most recent unstacked view

    prev_player_grids = set(get_occupied_grids_from_position(previous_frame.my_player.position))
    curr_player_grids = set(get_occupied_grids_from_position(current_frame.my_player.position))
    
    # Check the danger level at the player's current and previous grid positions
    # by sampling from the 11x11 danger_map centered on the player.
    center_y, center_x = 5, 5
    
    # Note: Since get_occupied_grids_from_position is complex, we simplify by just checking the center.
    # This is a reasonable approximation.
    prev_danger_val = danger_map[center_y, center_x]
    
    # We need to calculate the current danger value based on the *new* frame,
    # but using the *old* bomb locations for consistency.
    # A simpler way is to check the danger map at the new relative position.
    curr_pos = current_frame.my_player.position
    prev_pos = previous_frame.my_player.position
    dx = (curr_pos.x - prev_pos.x) // PIXEL_PER_CELL
    dy = (curr_pos.y - prev_pos.y) // PIXEL_PER_CELL
    
    curr_view_x, curr_view_y = center_x + dx, center_y + dy
    curr_danger_val = 0
    if 0 <= curr_view_y < VIEW_SIZE and 0 <= curr_view_x < VIEW_SIZE:
        curr_danger_val = danger_map[curr_view_y, curr_view_x]

    was_in_danger = prev_danger_val > 0
    is_in_danger = curr_danger_val > 0

    if not was_in_danger and is_in_danger:
        rewards['enter_danger_zone'] = -0.6
    elif was_in_danger and not is_in_danger:
        rewards['exit_danger_zone'] = 0.8

    # --- 12. Continuous Penalty for Staying in Danger Zone ---
    if is_in_danger:
        # Dynamic penalty based on the danger level
        # Cast to float for type consistency in the reward dictionary
        rewards['staying_in_danger'] = float(-0.7 * curr_danger_val)

        # --- 13. Reward for Moving Closer to a Safe Zone ---
        # This reward is critical. It solves the "U-turn" problem where the agent might
        # need to move through another dangerous square to reach the nearest safe exit.
        # It rewards any move that demonstrably reduces the distance to safety.
        if previous_action in [0, 1, 2, 3]: # If the agent tried to move
            # Use the observation the agent made its decision on
            grid_view_unstacked = previous_processed_obs['grid_view'][11:, :, :]
            
            # Calculate the complete safety distance map once
            safety_map = _calculate_safety_distance_map(grid_view_unstacked)

            # Previous position is the center of the view
            prev_view_y, prev_view_x = 5, 5
            dist_prev = safety_map[prev_view_y, prev_view_x]

            # Current position is offset from the center
            move_map = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)} # U, D, L, R
            dy, dx = move_map[previous_action]
            curr_view_y, curr_view_x = prev_view_y + dy, prev_view_x + dx
            
            # Ensure the new position is within the view before looking up the distance
            if 0 <= curr_view_y < VIEW_SIZE and 0 <= curr_view_x < VIEW_SIZE:
                dist_curr = safety_map[curr_view_y, curr_view_x]
                
                if dist_curr < dist_prev:
                    # This reward is set to 0.7 to perfectly counteract the maximum possible
                    # `staying_in_danger` penalty (-0.7 * 1.0), sending a clear signal
                    # that moving closer to safety is the correct action that neutralizes
                    # the immediate penalty of being in a dangerous spot.
                    rewards['moved_closer_to_safety'] = 0.8

    # --- 14. Reward for Moving Along the Shortest Path Gradient ---
    # This reward is now perfectly aligned with the observation, as it uses the
    # exact same grid_view that the agent used to make its decision.
    if previous_action in [0, 1, 2, 3] and not collided_with_wall:
        # 1. Get the gradient map that the agent saw
        # The passed `previous_processed_obs` is the stacked observation.
        # We need the most recent half, which is the second part.
        prev_grid_view = previous_processed_obs['grid_view']
        gradient_map = prev_grid_view[11:, :, :] # Get channels 11-21

        # 2. Find the best direction from the center (player's position)
        center_y, center_x = 5, 5
        current_gradient = gradient_map[10, center_y, center_x] # Channel 10 of the unstacked view
        
        best_direction = None
        max_gradient = current_gradient

        # Check neighbors in the 11x11 view
        for action, (dy, dx) in enumerate([( -1, 0), (1, 0), (0, -1), (0, 1)]): # U, D, L, R
            ny, nx = center_y + dy, center_x + dx
            
            neighbor_gradient = gradient_map[10, ny, nx]
            if neighbor_gradient > max_gradient:
                max_gradient = neighbor_gradient
                best_direction = action

        # 3. Reward if the chosen action matches the best direction
        if best_direction is not None and previous_action == best_direction:
            # This reward is increased to make exploration and territory capture
            # a more attractive strategy compared to passive or overly safe behavior.
            rewards['follow_gradient_path'] = 0.4

    return rewards
