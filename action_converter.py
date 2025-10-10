"""
动作空间转换工具
新旧动作空间的转换函数

速度说明：
- 游戏机制：stride=0时为最大速度16像素/tick，其他值为实际像素/tick
- 速度主要用于精确通过狭窄区域（如巷道），避免碰撞体积问题
"""
import numpy as np


# ============================================================================
# 旧动作空间 (Discrete(6)) → 新动作空间 (MultiDiscrete([5, 2, 5]))
# ============================================================================

def old_to_new_action(old_action: int) -> np.ndarray:
    """
    将旧的Discrete(6)动作转换为新的MultiDiscrete([5, 2, 5])动作
    
    旧动作:
        0: 上
        1: 下
        2: 左
        3: 右
        4: 放炸弹
        5: 停止
    
    新动作: [方向, 放炸弹, 速度]
        方向: 0=不动, 1=上, 2=下, 3=左, 4=右
        放炸弹: 0=不放, 1=放
        速度: 0=最大(16), 1=极慢(2), 2=慢(4), 3=中(8), 4=快(12)
    
    Args:
        old_action: 旧的单一动作值 (0-5)
    
    Returns:
        new_action: 新的动作数组 [direction, bomb, speed]
    """
    mapping = {
        0: [1, 0, 0],  # 上 → [上, 不放炸弹, 最大速度]
        1: [2, 0, 0],  # 下 → [下, 不放炸弹, 最大速度]
        2: [3, 0, 0],  # 左 → [左, 不放炸弹, 最大速度]
        3: [4, 0, 0],  # 右 → [右, 不放炸弹, 最大速度]
        4: [0, 1, 0],  # 放炸弹 → [不动, 放炸弹, 最大速度]
        5: [0, 0, 0],  # 停止 → [不动, 不放炸弹, 最大速度]
    }
    return np.array(mapping.get(old_action, [0, 0, 0]), dtype=np.int64)


# ============================================================================
# 新动作空间 (MultiDiscrete([5, 2, 5])) → 旧动作空间 (Discrete(6))
# ============================================================================

def new_to_old_action(new_action: np.ndarray) -> int:
    """
    将新的MultiDiscrete([5, 2, 5])动作转换回旧的Discrete(6)动作
    
    注意: 这是一个有损转换，因为新动作空间更丰富（速度信息会丢失）
    
    Args:
        new_action: 新的动作数组 [direction, bomb, speed]
    
    Returns:
        old_action: 旧的单一动作值 (0-5)
    """
    direction, bomb, speed = new_action
    
    # 优先判断是否放炸弹
    if bomb == 1:
        return 4  # 放炸弹
    
    # 根据方向返回
    direction_map = {
        0: 5,  # 不动 → 停止
        1: 0,  # 上
        2: 1,  # 下
        3: 2,  # 左
        4: 3,  # 右
    }
    return direction_map.get(direction, 5)


# ============================================================================
# 推断动作空间转换（用于数据收集）
# ============================================================================

def infer_multidiscrete_action(prev_player, curr_player, prev_bombs, curr_bombs, 
                               prev_frame=None, curr_frame=None) -> np.ndarray:
    """
    从前后帧推断MultiDiscrete动作（使用相对速度档位）
    
    Args:
        prev_player: 上一帧玩家状态
        curr_player: 当前帧玩家状态
        prev_bombs: 上一帧炸弹列表
        curr_bombs: 当前帧炸弹列表
        prev_frame: 上一帧完整数据（可选，用于更准确的速度推断）
        curr_frame: 当前帧完整数据（可选，用于检查加速点）
    
    Returns:
        action: [direction, bomb, speed]
            direction: 0=不动, 1=上, 2=下, 3=左, 4=右
            bomb: 0=不放, 1=放
            speed: 0=最大(100%), 1=极慢(20%), 2=慢(40%), 3=中(60%), 4=快(80%)
    """
    # 导入配置
    try:
        from config import BASE_SPEED, SPEED_PER_BOOT
    except:
        BASE_SPEED = 10.0
        SPEED_PER_BOOT = 2.0
    
    prev_x, prev_y = prev_player.position.x, prev_player.position.y
    curr_x, curr_y = curr_player.position.x, curr_player.position.y
    
    # 检查是否放了炸弹
    prev_bomb_count = sum(1 for b in prev_bombs if b.owner_id == curr_player.id)
    curr_bomb_count = sum(1 for b in curr_bombs if b.owner_id == curr_player.id)
    bomb = 1 if curr_bomb_count > prev_bomb_count else 0
    
    # 计算移动方向和距离
    dx = curr_x - prev_x
    dy = curr_y - prev_y
    distance = np.sqrt(dx**2 + dy**2)
    
    # 计算玩家的最大速度
    max_speed = BASE_SPEED + curr_player.agility_boots_count * SPEED_PER_BOOT
    
    # 检查是否踩在加速点上（如果提供了frame）
    if curr_frame is not None:
        player_pos = curr_player.position
        corners = [
            (player_pos.x - 25, player_pos.y - 25), (player_pos.x + 24, player_pos.y - 25),
            (player_pos.x - 25, player_pos.y + 24), (player_pos.x + 24, player_pos.y + 24)
        ]
        
        on_acceleration = False
        for corner_x, corner_y in corners:
            grid_x = int(corner_x / 50)
            grid_y = int(corner_y / 50)
            if 0 <= grid_y < 16 and 0 <= grid_x < 28:
                terrain = curr_frame.map[grid_y][grid_x].terrain
                if terrain == 'B':
                    on_acceleration = True
                    break
        
        if on_acceleration:
            max_speed *= 2.0
    
    # 推断速度档位（基于移动距离与最大速度的比例）
    if max_speed > 0:
        speed_ratio = distance / max_speed
        
        # 根据速度比例推断档位
        if speed_ratio > 0.9:  # 接近最大速度
            speed = 0  # 最大速度档位
        elif speed_ratio < 0.3:  # 很慢
            speed = 1  # 极慢档位(20%)
        elif speed_ratio < 0.5:
            speed = 2  # 慢档位(40%)
        elif speed_ratio < 0.7:
            speed = 3  # 中档位(60%)
        else:
            speed = 4  # 快档位(80%)
    else:
        # 回退到基于绝对距离的推断
        if distance < 3:
            speed = 1
        elif distance < 6:
            speed = 2
        elif distance < 10:
            speed = 3
        elif distance < 14:
            speed = 4
        else:
            speed = 0
    
    # 推断方向
    threshold = 5
    if abs(dy) > abs(dx):  # 垂直移动
        if dy < -threshold:
            direction = 1  # 上
        elif dy > threshold:
            direction = 2  # 下
        else:
            direction = 0  # 不动
    else:  # 水平移动
        if dx < -threshold:
            direction = 3  # 左
        elif dx > threshold:
            direction = 4  # 右
        else:
            direction = 0  # 不动
    
    return np.array([direction, bomb, speed], dtype=np.int64)


# ============================================================================
# 动作描述
# ============================================================================

def describe_action(action: np.ndarray, max_speed: float = None) -> str:
    """
    将动作数组转换为可读的描述
    
    Args:
        action: [direction, bomb, speed]
        max_speed: 玩家当前最大速度（可选，用于显示实际速度）
    
    Returns:
        description: 动作描述字符串
    """
    direction_names = ["不动", "上", "下", "左", "右"]
    bomb_names = ["", "+放炸弹"]
    speed_names = ["最大(100%)", "极慢(20%)", "慢速(40%)", "中速(60%)", "快速(80%)"]
    speed_percentages = [1.0, 0.2, 0.4, 0.6, 0.8]
    
    direction = direction_names[action[0]]
    bomb = bomb_names[action[1]]
    speed_name = speed_names[action[2]]
    speed_pct = speed_percentages[action[2]]
    
    # 如果提供了最大速度，计算实际速度
    if max_speed is not None:
        if action[2] == 0:
            actual_speed = max_speed
            speed_desc = f"{speed_name}, {actual_speed:.1f}px/tick"
        else:
            actual_speed = max_speed * speed_pct
            speed_desc = f"{speed_name}, ~{actual_speed:.1f}px/tick"
    else:
        speed_desc = speed_name
    
    if action[0] == 0 and action[1] == 0:
        return f"停止 ({speed_desc})"
    elif action[0] == 0:
        return f"原地放炸弹 ({speed_desc})"
    else:
        return f"{direction}{bomb} ({speed_desc})"


# ============================================================================
# 批量转换
# ============================================================================

def convert_old_actions_batch(old_actions: np.ndarray) -> np.ndarray:
    """
    批量转换旧动作到新动作
    
    Args:
        old_actions: shape (N,) 旧动作数组
    
    Returns:
        new_actions: shape (N, 3) 新动作数组
    """
    return np.array([old_to_new_action(a) for a in old_actions])


def convert_new_actions_batch(new_actions: np.ndarray) -> np.ndarray:
    """
    批量转换新动作到旧动作
    
    Args:
        new_actions: shape (N, 3) 新动作数组
    
    Returns:
        old_actions: shape (N,) 旧动作数组
    """
    return np.array([new_to_old_action(a) for a in new_actions])


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    print("动作空间转换示例\n")
    
    # 旧 → 新
    print("旧动作 → 新动作:")
    for old_act in range(6):
        new_act = old_to_new_action(old_act)
        description = describe_action(new_act)
        print(f"  {old_act} → {new_act} ({description})")
    
    print("\n新动作示例 (假设玩家有3个鞋子，max_speed=16):")
    test_actions = [
        [1, 0, 0],  # 向上，不放炸弹，最大速度
        [1, 1, 4],  # 向上，放炸弹，快速
        [0, 1, 3],  # 原地，放炸弹，中速
        [4, 0, 1],  # 向右，不放炸弹，极慢速（精确控制）
        [3, 0, 2],  # 向左，不放炸弹，慢速（通过狭窄区域）
    ]
    
    # 示例：3个鞋子的玩家
    max_speed_example = 16.0  # 10 + 3*2
    
    for action in test_actions:
        action_arr = np.array(action)
        description = describe_action(action_arr, max_speed_example)
        old_action = new_to_old_action(action_arr)
        print(f"  {action} → {description} (旧动作: {old_action})")
    
    print("\n踩加速点时 (max_speed=32):")
    max_speed_boosted = 32.0
    for action in [[1, 0, 0], [4, 0, 2]]:
        action_arr = np.array(action)
        description = describe_action(action_arr, max_speed_boosted)
        print(f"  {action} → {description}")

