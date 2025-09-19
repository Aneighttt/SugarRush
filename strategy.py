"""
AI Strategy - 策略/决策层
AI的“大脑”，负责制定具体行动。
它使用 ai_core.py 中的工具来做出决策。
"""
from typing import List, Tuple, Optional, Dict, Any
from data_models import Frame
from ai_core import GameMap, DangerEvaluator, AStarPathfinder
import numpy as np
class AIStrategy:
    """
    封装了AI的所有决策逻辑。
    """
    def __init__(self, map_width: int, map_height: int, grid_size: int):
        """
        初始化策略所需的核心组件。
        """
        self.game_map = GameMap(map_width, map_height, grid_size)
        self.danger_evaluator = DangerEvaluator(self.game_map)
        self.pathfinder = AStarPathfinder(self.game_map)
        
        # 存储AI自身的状态
        self.my_player_info = None
        self.my_bottom_left_pos: Optional[Tuple[int, int]] = None

    def make_decision(self, frame: Frame) -> Optional[Dict[str, Any]]:
        """
        AI的主决策函数。
        接收服务器的 frame 数据，并返回一个行动指令字典或None。
        """
        # 1. 更新世界状态
        self.game_map.update_from_frame(frame)
        
        all_players = frame.other_players + ([frame.my_player] if frame.my_player else [])
        self.danger_evaluator.update(frame.bombs, all_players, frame.current_tick)
        
        self.my_player_info = frame.my_player
        if not self.my_player_info or not self.my_player_info.position:
            self.my_bottom_left_pos = None
            return None # 没有玩家信息，无法决策
        
        # 根据玩家中点位置，计算左下角判定点位置
        self.my_bottom_left_pos = (self.my_player_info.position.x - 25, self.my_player_info.position.y - 25)

        # 2. 决策逻辑 (分层决策)
        current_tick = frame.current_tick
        
        # 最高优先级：生存
        if self._is_in_danger(current_tick):
            action = self._evade_danger(current_tick)
            if action:
                return action

        # 第二优先级：执行战术 (例如：炸墙)
        # tactical_action = self._execute_tactic()
        # if tactical_action:
        #     return tactical_action

        # 最低优先级：战略移动 (例如：探索)
        # move_action = self._strategic_move()
        # if move_action:
        #     return move_action
        
        # 默认不执行任何操作
        return None

    def _is_in_danger(self, current_tick: int) -> bool:
        """检查自己当前位置是否危险。"""
        if not self.my_bottom_left_pos:
            return False
        
        my_grid_pos = (self.my_bottom_left_pos[0] // self.game_map.grid_size, 
                       self.my_bottom_left_pos[1] // self.game_map.grid_size)
        
        return self.danger_evaluator.is_grid_dangerous(my_grid_pos[0], my_grid_pos[1], current_tick)

    def _evade_danger(self, current_tick: int) -> Optional[Dict[str, Any]]:
        """寻找并移动到最近的安全点。"""
        print(f"[Tick: {current_tick}] DANGER DETECTED!")
        my_pos = self.my_bottom_left_pos
        print(f"    - Current Position: {my_pos}")

        safe_spots = self._find_safe_spots(current_tick)
        if not safe_spots:
            print("    - NO SAFE SPOTS FOUND!")
            return None

        closest_spot = min(safe_spots, key=lambda spot: 
            abs(spot[0] * self.game_map.grid_size - my_pos[0]) + 
            abs(spot[1] * self.game_map.grid_size - my_pos[1]))
        
        # 修复: 目标位置应为目标格子的左下角
        target_pixel_pos = (closest_spot[0] * self.game_map.grid_size,
                            closest_spot[1] * self.game_map.grid_size)
        print(f"    - Target Position: {target_pixel_pos} (Grid: {closest_spot})")

        # 添加: 打印当前格子和目标格子的微观视图
        current_grid_pos = (my_pos[0] // self.game_map.grid_size, my_pos[1] // self.game_map.grid_size)
        print(f"--- Walkability for Current Grid {current_grid_pos} ---")
        self._visualize_grid_walkability(current_grid_pos)
        print(f"--- Walkability for Target Grid {closest_spot} ---")
        self._visualize_grid_walkability(closest_spot)

        path = self.pathfinder.find_path(my_pos, target_pixel_pos, current_tick, self.danger_evaluator)

        if path and len(path) > 1:
            print(f"    - Path FOUND! Length: {len(path)}")
            print(f"        - Path: {path}")
            self._visualize_map(my_pos, target_pixel_pos, path)
            return self._path_to_command(path)
        else:
            print("    - Path NOT FOUND!")
            self._visualize_map(my_pos, target_pixel_pos, None)
            return None

    def _path_to_command(self, path: List[Tuple[int, int]]) -> Optional[Dict[str, Any]]:
        """将像素路径转换为服务器可以理解的移动指令。"""
        if not self.my_player_info or not self.my_bottom_left_pos or len(path) < 2:
            return None

        my_pos = self.my_bottom_left_pos
        target_node_index = min(len(path) - 1, 1)
        next_pos = path[target_node_index]

        dx = next_pos[0] - my_pos[0]
        dy = next_pos[1] - my_pos[1]

        # 确定主导移动方向, 并映射到服务器的枚举
        if abs(dx) > abs(dy):
            direction = "R" if dx > 0 else "L" # Right / Left
        else:
            direction = "U" if dy > 0 else "D" # Up / Down
        
        # 根据公式精确计算速度
        speed = 10 + 2 * self.my_player_info.agility_boots_count
        
        # 移动距离不能超过到下一个路径点的距离
        distance_to_next = (dx**2 + dy**2)**0.5
        stride = min(speed, distance_to_next)

        return {
            "direction": direction,
            "is_place_bomb": False,
            "stride": int(round(stride)) # stride 必须是整数
        }

    def _find_safe_spots(self, current_tick: int) -> List[Tuple[int, int]]:
        """遍历地图，找到所有当前不是危险区的空格子。"""
        safe_spots = []
        for internal_gx in range(self.game_map.width_grid):
            for internal_gy in range(self.game_map.height_grid):
                # 必须是空地
                if self.game_map.grid[internal_gx, internal_gy] == 0:
                    # 将内部格子坐标转换为服务器坐标以查询危险
                    server_gy = self.game_map.height_grid - 1 - internal_gy
                    server_gx = internal_gx
                    
                    if not self.danger_evaluator.is_grid_dangerous(server_gx, server_gy, current_tick):
                        # 存储服务器坐标系下的安全点
                        safe_spots.append((server_gx, server_gy))
        return safe_spots

    def _visualize_grid_walkability(self, grid_pos: Tuple[int, int]) -> None:
        """打印指定格子内部50x50像素的可行走图。"""
        gx, gy = grid_pos
        start_px = gx * self.game_map.grid_size
        start_py = gy * self.game_map.grid_size
        
        # 注意: 我们从上到下打印，所以Y轴是反的
        for y_offset in range(self.game_map.grid_size -1, -1, -1):
            row_str = ""
            for x_offset in range(self.game_map.grid_size):
                px = start_px + x_offset
                py = start_py + y_offset
                if self.game_map.is_pixel_walkable(px, py):
                    row_str += ". " # Walkable
                else:
                    row_str += "X " # Not Walkable
            print(row_str)

    def _visualize_map(self, player_pos: Tuple[int, int], target_pos: Tuple[int, int], path: Optional[List[Tuple[int, int]]]) -> None:
        """在控制台打印可视化地图。"""
        from ai_core import HARD_WALL, SOFT_WALL
        
        # 1. 创建一个基于字符的地图副本
        grid_map = np.full((self.game_map.width_grid, self.game_map.height_grid), ' ', dtype=str)
        for gx in range(self.game_map.width_grid):
            for gy in range(self.game_map.height_grid):
                if self.game_map.grid[gx, gy] == HARD_WALL:
                    grid_map[gx, gy] = '#'
                elif self.game_map.grid[gx, gy] == SOFT_WALL:
                    grid_map[gx, gy] = '%'

        # 2. 标记路径
        if path:
            for pos in path:
                gx, gy = pos[0] // self.game_map.grid_size, pos[1] // self.game_map.grid_size
                internal_gy = self.game_map.height_grid - 1 - gy
                if 0 <= gx < self.game_map.width_grid and 0 <= internal_gy < self.game_map.height_grid:
                    if grid_map[gx, internal_gy] == ' ':
                        grid_map[gx, internal_gy] = '*'

        # 3. 标记炸弹
        for bomb_pos_grid in self.danger_evaluator.danger_map.keys():
            gx, gy = bomb_pos_grid
            internal_gy = self.game_map.height_grid - 1 - gy
            if 0 <= gx < self.game_map.width_grid and 0 <= internal_gy < self.game_map.height_grid:
                grid_map[gx, internal_gy] = 'B'

        # 4. 标记玩家和目标
        player_gx, player_gy = player_pos[0] // self.game_map.grid_size, player_pos[1] // self.game_map.grid_size
        internal_player_gy = self.game_map.height_grid - 1 - player_gy
        grid_map[player_gx, internal_player_gy] = 'P'

        target_gx, target_gy = target_pos[0] // self.game_map.grid_size, target_pos[1] // self.game_map.grid_size
        internal_target_gy = self.game_map.height_grid - 1 - target_gy
        grid_map[target_gx, internal_target_gy] = 'T'

        # 5. 打印地图 (需要转置并翻转Y轴以匹配控制台输出习惯)
        print("--- MAP VISUALIZATION ---")
        # Transpose to get (y, x) and then flip y-axis to print top-to-bottom
        transposed_map = grid_map.T
        # 修复: 格式化输出，确保每个格子宽度相同，使其对齐
        for row in transposed_map:
            print("".join([f"{c:<2}" for c in row]))
        print("-------------------------")
