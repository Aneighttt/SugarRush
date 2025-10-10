"""
多视角观察处理器
支持从任意玩家的视角创建观察
用于收集专家AI数据
"""
import numpy as np
from data_models import Frame, Player
from config import *
from utils import visualize_gradient_in_terminal, find_path_to_nearest_frontier
from frame_processor import pixels_to_grid


def create_grid_view_for_player(frame: Frame, target_player: Player) -> np.ndarray:
    """
    从指定玩家的视角创建grid_view
    
    Args:
        frame: 游戏帧
        target_player: 目标玩家（可以是my_player或other_players中的任意一个）
    
    Returns:
        grid_view: (14, 16, 28)的numpy数组
    """
    current_tick = frame.current_tick
    grid_view = np.zeros((MAP_CHANNELS, MAP_HEIGHT, MAP_WIDTH), dtype=np.float32)
    
    # Channel 0: Terrain
    for y in range(MAP_HEIGHT):
        for x in range(MAP_WIDTH):
            cell = frame.map[y][x]
            if cell.terrain == 'D':
                grid_view[0, y, x] = 0.5
            elif cell.terrain in ['I', 'N']:
                grid_view[0, y, x] = 1.0

    # Channel 1: Bombs
    for bomb in frame.bombs:
        bx, by = pixels_to_grid(bomb.position.x, bomb.position.y)
        if 0 <= by < MAP_HEIGHT and 0 <= bx < MAP_WIDTH:
            grid_view[1, by, bx] = 1.0

    # Channel 2: Danger Zone
    all_explosion_chains = []
    if frame.bombs:
        bombs_by_pos = {(b.position.x, b.position.y): b for b in frame.bombs}
        unvisited_bombs = set(frame.bombs)
        
        while unvisited_bombs:
            q = [unvisited_bombs.pop()]
            chain = {'bombs': set(q), 'danger_value': 0}
            head = 0
            
            while head < len(q):
                bomb = q[head]
                head += 1
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    for i in range(1, bomb.range + 1):
                        cbx, cby = pixels_to_grid(bomb.position.x, bomb.position.y)
                        nx, ny = cbx + dx * i, cby + dy * i
                        if not (0 <= ny < MAP_HEIGHT and 0 <= nx < MAP_WIDTH):
                            break
                        npx, npy = nx * 50 + 25, ny * 50 + 25
                        if (npx, npy) in bombs_by_pos:
                            chained_bomb = bombs_by_pos[(npx, npy)]
                            if chained_bomb in unvisited_bombs:
                                unvisited_bombs.remove(chained_bomb)
                                chain['bombs'].add(chained_bomb)
                                q.append(chained_bomb)
                        if frame.map[ny][nx].terrain in ['I', 'N', 'D']:
                            break
            
            first_tick = min(b.explode_at for b in chain['bombs'])
            time_to_explosion = max(0.0, first_tick - current_tick)
            chain['danger_value'] = 1.0 / (1.0 + time_to_explosion)
            all_explosion_chains.append(chain)

            for bomb in chain['bombs']:
                bx, by = pixels_to_grid(bomb.position.x, bomb.position.y)
                if 0 <= by < MAP_HEIGHT and 0 <= bx < MAP_WIDTH:
                    grid_view[2, by, bx] = max(grid_view[2, by, bx], chain['danger_value'])
                    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        for i in range(1, bomb.range + 1):
                            nx, ny = bx + dx * i, by + dy * i
                            if not (0 <= ny < MAP_HEIGHT and 0 <= nx < MAP_WIDTH):
                                break
                            grid_view[2, ny, nx] = max(grid_view[2, ny, nx], chain['danger_value'])
                            if frame.map[ny][nx].terrain in ['I', 'N', 'D']:
                                break

    # Channels 3, 4, 5: Items
    for item in frame.map_items:
        ix, iy = pixels_to_grid(item.position.x, item.position.y)
        if 0 <= iy < MAP_HEIGHT and 0 <= ix < MAP_WIDTH:
            if item.type == 'AB':
                grid_view[3, iy, ix] = 1.0
            elif item.type == 'SP':
                grid_view[4, iy, ix] = 1.0
            elif item.type == 'BP':
                grid_view[5, iy, ix] = 1.0

    # Channels 6-9: Special Terrains and Occupied Zones
    for y in range(MAP_HEIGHT):
        for x in range(MAP_WIDTH):
            cell = frame.map[y][x]
            if cell.terrain in ['I', 'N', 'D']:
                continue
            if cell.terrain == 'B':
                grid_view[6, y, x] = 1.0
            elif cell.terrain == 'M':
                grid_view[7, y, x] = 1.0
            
            if cell.ownership != 'N' and cell.ownership != target_player.team:
                grid_view[8, y, x] = 1.0  # Enemy occupied
            elif cell.ownership == 'N':
                grid_view[9, y, x] = 1.0  # Non occupied

    # Channel 10: Gradient field (针对target_player计算)
    # 这里需要修改find_path_to_nearest_frontier函数以支持指定player
    # 暂时使用原来的逻辑
    reconstructed_path, target = find_path_to_nearest_frontier(frame)
    if target:
        path_len = len(reconstructed_path)
        if path_len > 1:
            for i, (py, px) in enumerate(reconstructed_path):
                if 0 <= py < MAP_HEIGHT and 0 <= px < MAP_WIDTH:
                    grid_view[10, py, px] = 1.0
        elif path_len == 1:
            py, px = reconstructed_path[0]
            if 0 <= py < MAP_HEIGHT and 0 <= px < MAP_WIDTH:
                grid_view[10, py, px] = 1.0
        ty, tx = target
        if 0 <= ty < MAP_HEIGHT and 0 <= tx < MAP_WIDTH:
            grid_view[10, ty, tx] = 1.0

    # Channel 11: target_player自己的位置
    my_px, my_py = pixels_to_grid(target_player.position.x, target_player.position.y)
    if 0 <= my_py < MAP_HEIGHT and 0 <= my_px < MAP_WIDTH:
        grid_view[11, my_py, my_px] = 1.0

    # Channel 12: 队友位置
    # Channel 13: 敌人位置
    # 获取所有其他玩家（不包括target_player）
    all_players = [frame.my_player] + frame.other_players
    for player in all_players:
        if player.id == target_player.id:
            continue  # 跳过自己
        
        op_px, op_py = pixels_to_grid(player.position.x, player.position.y)
        if 0 <= op_py < MAP_HEIGHT and 0 <= op_px < MAP_WIDTH:
            if player.team == target_player.team:
                grid_view[12, op_py, op_px] = 1.0  # 队友
            else:
                grid_view[13, op_py, op_px] = 1.0  # 敌人

    return grid_view


def create_player_state_for_player(frame: Frame, target_player: Player) -> np.ndarray:
    """
    为指定玩家创建player_state
    
    Args:
        frame: 游戏帧
        target_player: 目标玩家
    
    Returns:
        player_state: (8,)的numpy数组
    """
    # 计算当前炸弹数量
    current_bomb_count = sum(1 for bomb in frame.bombs if bomb.owner_id == target_player.id)
    
    # 炸弹范围
    bomb_range = target_player.sweet_potion_count * RANGE_PER_POTION
    
    # 基础速度
    speed = BASE_SPEED + target_player.agility_boots_count * SPEED_PER_BOOT
    
    # 检查地形速度修正
    player_pos = target_player.position
    corners = [
        (player_pos.x - 25, player_pos.y - 25), (player_pos.x + 24, player_pos.y - 25),
        (player_pos.x - 25, player_pos.y + 24), (player_pos.x + 24, player_pos.y + 24)
    ]
    
    on_acceleration = False
    on_deceleration = False
    for corner_x, corner_y in corners:
        grid_x, grid_y = pixels_to_grid(corner_x, corner_y)
        if 0 <= grid_y < MAP_HEIGHT and 0 <= grid_x < MAP_WIDTH:
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
    
    normalized_bomb_range = np.clip(bomb_range, 0, MAX_RANGE) / MAX_RANGE
    
    # 位置偏移
    px, py = target_player.position.x, target_player.position.y
    gx, gy = pixels_to_grid(px, py)
    center_x, center_y = gx * 50 + 25, gy * 50 + 25
    offset_x, offset_y = (px - center_x) / 50, (py - center_y) / 50
    offset_x *= 2
    offset_y *= 2
    
    # 检查extra_status的三种状态及剩余时间：INV(无敌), THB(加速), LIT(减速)
    # status总生效时间300tick，剩余时间归一化到[0,1]
    MAX_STATUS_DURATION = 300.0
    
    inv_remaining = 0.0
    thb_remaining = 0.0
    lit_remaining = 0.0
    
    if target_player.extra_status:
        current_tick = frame.current_tick
        for s in target_player.extra_status:
            remaining_ticks = max(0.0, s.expire_at - current_tick)
            normalized_remaining = min(1.0, remaining_ticks / MAX_STATUS_DURATION)
            
            if s.name == 'INV':
                inv_remaining = normalized_remaining
            elif s.name == 'THB':
                thb_remaining = normalized_remaining
            elif s.name == 'LIT':
                lit_remaining = normalized_remaining
    
    return np.array([
        offset_x,
        offset_y,
        target_player.bomb_pack_count / MAX_BOMB_PACK,
        normalized_bomb_range,
        speed / MAX_SPEED,
        1.0 if current_bomb_count < target_player.bomb_pack_count else 0.0,
        1.0 if target_player.status == 'D' else 0.0,
        inv_remaining,  # INV剩余时间（归一化到[0,1]）
        thb_remaining,  # THB剩余时间（归一化到[0,1]）
        lit_remaining,  # LIT剩余时间（归一化到[0,1]）
    ], dtype=np.float32)


def preprocess_observation_for_player(frame: Frame, target_player: Player) -> dict:
    """
    为指定玩家创建完整观察
    
    Args:
        frame: 游戏帧
        target_player: 目标玩家
    
    Returns:
        observation: 字典格式的观察
    """
    grid_view = create_grid_view_for_player(frame, target_player)
    player_state = create_player_state_for_player(frame, target_player)
    
    return {
        "grid_view": grid_view,
        "player_state": player_state
    }

