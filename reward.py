from data_models import Frame
from utils import MAP_WIDTH, MAP_HEIGHT

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
    PIXEL_PER_CELL = 50
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
        
    return explosion_grids

def get_danger_zones(frame: Frame):
    danger_zones = set()
    for bomb in frame.bombs:
        danger_zones.update(_get_explosion_grids(bomb, frame))
    return danger_zones

def calculate_reward(current_frame: Frame, previous_frame: Frame, prev_info: dict, previous_action: int):
    reward = 0
    current_bombs = [b for b in current_frame.bombs if b.owner_id == current_frame.my_player.id]
    
    current_my_territory, current_enemy_territory = count_territory(current_frame)
    prev_diff = prev_info['my_territory'] - prev_info['enemy_territory']
    current_diff = current_my_territory - current_enemy_territory
    reward += (current_diff - prev_diff) * 0.5 # Scaled down

    if current_frame.current_tick == 1800:
        if current_my_territory > current_enemy_territory:
            reward += 3 # Scaled down
        elif current_my_territory < current_enemy_territory:
            reward -= 3 # Scaled down

    is_currently_stunned = (current_frame.my_player.status == 'D')
    if is_currently_stunned and not prev_info['is_stunned']:
        reward -= 20 # Scaled down

    current_items = current_frame.my_player.bomb_pack_count + current_frame.my_player.sweet_potion_count + current_frame.my_player.agility_boots_count
    if current_items > prev_info['items_collected']:
        reward += 5 # Scaled down

    if previous_action == 4:
        current_bomb_identifiers = {(b.position.x, b.position.y, b.explode_at) for b in current_bombs}
        prev_bomb_identifiers = prev_info.get('my_bomb_identifiers', set())
        
        new_bomb_identifiers = current_bomb_identifiers - prev_bomb_identifiers
        if new_bomb_identifiers:
            new_bomb_identifier = new_bomb_identifiers.pop()
            new_bomb = next((b for b in current_bombs if (b.position.x, b.position.y, b.explode_at) == new_bomb_identifier), None)
            
            if new_bomb:
                strategic_value = 0
                explosion_area = _get_explosion_grids(new_bomb, current_frame)
                enemy_grids = set()
                my_grids = set(get_occupied_grids_from_position(current_frame.my_player.position))

                for p in current_frame.other_players:
                    if p.team != current_frame.my_player.team:
                        enemy_grids.update(get_occupied_grids_from_position(p.position))
                
                if my_grids.intersection(explosion_area):
                    strategic_value -= 10 # Scaled down

                for x, y in explosion_area:
                    if current_frame.map[y][x].terrain == 'D':
                        strategic_value += 0.2 # Scaled down
                    if (x, y) in enemy_grids:
                        strategic_value += 3 # Scaled down
                
                reward += strategic_value

    is_move_action = previous_action in [0, 1, 2, 3]
    if is_move_action and current_frame.my_player.position == previous_frame.my_player.position:
        reward -= 0.5 # Scaled down

    if previous_action == 5:
        reward -= 1 # Scaled down

    reward -= 0.001 # Small penalty for existing

    player_grids = get_occupied_grids_from_position(current_frame.my_player.position)
    danger_zones = get_danger_zones(current_frame)
    if any(grid in danger_zones for grid in player_grids):
        reward -= 0.5 # Scaled down

    last_action = prev_info.get('last_action', -1)
    current_action = previous_action
    opposites = {0: 1, 1: 0, 2: 3, 3: 2}
    if opposites.get(last_action) == current_action:
        reward -= 0.2 # Scaled down

    new_info = {
        'my_territory': current_my_territory,
        'enemy_territory': current_enemy_territory,
        'is_stunned': is_currently_stunned,
        'items_collected': current_items,
        'my_bomb_identifiers': {(b.position.x, b.position.y, b.explode_at) for b in current_bombs},
        'last_action': previous_action
    }
    return reward, new_info
