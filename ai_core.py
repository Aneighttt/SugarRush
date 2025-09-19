"""
AI Core - 核心工具/算法层
包含所有与具体“策略”无关的、可重用的核心组件。
"""

from typing import List, Tuple, Optional
from data_models import Frame, Bomb, Player
import numpy as np

# 定义地图元素常量
EMPTY = 0
SOFT_WALL = 1
HARD_WALL = 2

class GameMap:
    """
    管理和解析游戏地图信息。
    - 解析服务器的 frame 数据。
    - 维护一个二维网格来表示世界。
    - 处理障碍物的体积膨胀，为路径规划提供可行走区域。
    """
    def __init__(self, width_px: int, height_px: int, grid_size: int):
        self.width_px = width_px
        self.height_px = height_px
        self.grid_size = grid_size
        self.width_grid = width_px // grid_size
        self.height_grid = height_px // grid_size
        
        # 彻底修复: 所有内部数据结构都使用服务器坐标系 (左下角为原点)
        # grid 形状为 (width, height)
        self.grid = np.full((self.width_grid, self.height_grid), EMPTY, dtype=int)
        # walkable_map 形状为 (width, height)
        self.walkable_map = np.full((self.width_px, self.height_px), True, dtype=bool)
        self._map_dirty = True

    def update_from_frame(self, frame: Frame) -> None:
        """从服务器的 frame 数据更新地图状态。"""
        self.grid.fill(EMPTY)
        if frame.map:
            # frame.map 的索引是 [y][x], 直接使用
            for gy, row in enumerate(frame.map):
                for gx, cell in enumerate(row):
                    if cell.terrain == 'D':
                        self.grid[gx, gy] = SOFT_WALL
                    elif cell.terrain in ['I', 'N']:
                        self.grid[gx, gy] = HARD_WALL
        
        self._map_dirty = True
        self._update_walkable_map()

    def _update_walkable_map(self):
        """
        根据当前的 grid 状态，重新计算整个像素级的碰撞地图。
        所有计算都在服务器坐标系下进行，不再需要转换。
        """
        if not self._map_dirty:
            return
            
        self.walkable_map.fill(True)
        EXPANSION = self.grid_size - 1

        # 1. 地图边界膨胀
        self.walkable_map[self.width_px - EXPANSION:, :] = False
        self.walkable_map[:, self.height_px - EXPANSION:] = False

        # 2. 障碍物膨胀
        for gx in range(self.width_grid):
            for gy in range(self.height_grid):
                if self.grid[gx, gy] != EMPTY:
                    x_min = gx * self.grid_size - EXPANSION
                    x_max = (gx + 1) * self.grid_size - 1
                    y_min = gy * self.grid_size - EXPANSION
                    y_max = (gy + 1) * self.grid_size - 1

                    clip_x_start = max(0, x_min)
                    clip_x_end = min(self.width_px, x_max + 1)
                    clip_y_start = max(0, y_min)
                    clip_y_end = min(self.height_px, y_max + 1)
                    
                    if clip_x_start < clip_x_end and clip_y_start < clip_y_end:
                        self.walkable_map[clip_x_start:clip_x_end, clip_y_start:clip_y_end] = False
        
        self._map_dirty = False

    def is_pixel_walkable(self, x: int, y: int) -> bool:
        """
        判断一个像素点是否是可走的，已经考虑了障碍物的体积膨胀。
        """
        if not (0 <= x < self.width_px and 0 <= y < self.height_px):
            return False
        
        is_walkable = self.walkable_map[int(x), int(y)]
            
        return is_walkable


class DangerEvaluator:
    """
    评估地图的危险区域。
    """
    def __init__(self, game_map: GameMap):
        self.game_map = game_map
        # danger_map 存储每个危险格子的爆炸时间
        # 格式: {(gx, gy): explosion_time}
        self.danger_map = {}

    def update(self, bombs: List[Bomb], players: List[Player], current_tick: int) -> None:
        """
        根据当前所有的炸弹，更新危险地图。
        :param bombs: 服务器 frame 中的炸弹列表
        :param players: 所有玩家的列表，用于查找炸弹威力
        :param current_tick: 当前的游戏 tick 或帧数
        """
        if(bombs is None):
            return
        self.danger_map.clear()
        grid_size = self.game_map.grid_size

        for bomb in bombs:
            explosion_time = bomb.explode_at
            if not bomb.position or explosion_time is None:
                continue
            # 修复: 根据用户反馈, bomb.position 是格子坐标, 直接使用
            gx, gy = bomb.position.x, bomb.position.y
            
            # 使用炸弹自身的 range 属性
            power = bomb.range

            # 将炸弹本身所在格子标记为危险
            if (gx, gy) not in self.danger_map or self.danger_map[(gx, gy)] > explosion_time:
                self.danger_map[(gx, gy)] = explosion_time

            # 向四个方向蔓延
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            for dx, dy in directions:
                for i in range(1, power + 1):
                    nx, ny = gx + dx * i, gy + dy * i

                    # 检查是否越界
                    if not (0 <= nx < self.game_map.width_grid and 0 <= ny < self.game_map.height_grid):
                        break
                    
                    # 标记危险区域
                    if (nx, ny) not in self.danger_map or self.danger_map[(nx, ny)] > explosion_time:
                        self.danger_map[(nx, ny)] = explosion_time

                    # grid 现在是服务器坐标系，直接检查
                    if self.game_map.grid[nx, ny] != EMPTY:
                        break

    def is_grid_dangerous(self, gx: int, gy: int, current_tick: int) -> bool:
        """
        查询某个格子在当前或未来是否危险。
        """
        if (gx, gy) in self.danger_map:
            # 如果爆炸时间大于当前时间，说明危险仍然存在
            return self.danger_map[(gx, gy)] > current_tick
        return False


import heapq

class AStarPathfinder:
    """
    A* 路径规划器，负责在像素级别上寻找最优路径。
    """
    def __init__(self, game_map: GameMap, step_size: int = 10):
        self.game_map = game_map
        self.step_size = step_size # 每次扩展节点的像素步长
        # 修复: 只能上下左右移动
        self.directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """计算两个像素点之间的曼哈顿距离作为启发函数。"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _is_body_safe(self, bottom_left_pos: Tuple[int, int], current_tick: int, danger_evaluator: Optional[DangerEvaluator]) -> bool:
        """检查机器人的整个身体在指定位置是否安全。"""
        BODY_SIZE = 49
        
        bl = bottom_left_pos
        br = (bl[0] + BODY_SIZE, bl[1])
        tl = (bl[0], bl[1] + BODY_SIZE)
        tr = (bl[0] + BODY_SIZE, bl[1] + BODY_SIZE)
        
        corners = [bl, br, tl, tr]
        
        for corner in corners:
            # 检查碰撞
            if not self.game_map.is_pixel_walkable(corner[0], corner[1]):
                return False
            
            # 检查危险
            if danger_evaluator:
                gx = corner[0] // self.game_map.grid_size
                gy = corner[1] // self.game_map.grid_size
                if danger_evaluator.is_grid_dangerous(gx, gy, current_tick):
                    return False
                    
        return True

    def find_path(self, start_pos: Tuple[int, int], end_pos: Tuple[int, int], current_tick: int = 0, danger_evaluator: Optional[DangerEvaluator] = None) -> Optional[List[Tuple[int, int]]]:
        """
        寻找从起点像素到终点像素的路径。
        """
        start_pos = (int(start_pos[0]), int(start_pos[1]))
        end_pos = (int(end_pos[0]), int(end_pos[1]))

        open_set = []
        heapq.heappush(open_set, (0, start_pos)) # (f_cost, pos)
        
        came_from = {}
        g_cost = {start_pos: 0}

        while open_set:
            current_f, current_pos = heapq.heappop(open_set)

            if self._heuristic(current_pos, end_pos) < self.step_size:
                # 已经非常接近终点，构建路径
                path = []
                while current_pos in came_from:
                    path.append(current_pos)
                    current_pos = came_from[current_pos]
                path.append(start_pos)
                return path[::-1]

            for dx, dy in self.directions:
                neighbor_pos = (current_pos[0] + dx * self.step_size, 
                                current_pos[1] + dy * self.step_size)

                # 1. 碰撞检测
                if not self.game_map.is_pixel_walkable(neighbor_pos[0], neighbor_pos[1]):
                    continue

                # 2. 危险区域检测 (简化版)
                if danger_evaluator:
                    gx = neighbor_pos[0] // self.game_map.grid_size
                    gy = neighbor_pos[1] // self.game_map.grid_size
                    if danger_evaluator.is_grid_dangerous(gx, gy, current_tick):
                        continue
                
                # 3. 计算成本 (现在只有直线移动)
                new_g = g_cost[current_pos] + self.step_size
                
                if neighbor_pos not in g_cost or new_g < g_cost[neighbor_pos]:
                    g_cost[neighbor_pos] = new_g
                    f_cost = new_g + self._heuristic(neighbor_pos, end_pos)
                    heapq.heappush(open_set, (f_cost, neighbor_pos))
                    came_from[neighbor_pos] = current_pos
        
        return None # 未找到路径
