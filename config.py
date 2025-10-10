# ==============================================================================
# FILE: config.py
# PURPOSE: Central configuration file for shared constants to avoid circular imports.
# ==============================================================================

# --- Debugging Flags ---
# Set to True to print the pathfinding gradient map to the console for each frame.
# This will significantly slow down the process and spam the console.
DEBUG_VISUALIZE_GRADIENT = False

# Set to True to print the calculated danger zone map (grid_view channel 2) to the console.
DEBUG_DANGER_ZONE = False

# --- Grid and View Dimensions ---
MAP_WIDTH = 28
MAP_HEIGHT = 16
VIEW_SIZE = 11
# Updated to match the current implementation (Channels 0-13)
MAP_CHANNELS = 14
PLAYER_STATE_SIZE = 10  # 更新为10维（新增THB和LIT状态）
PIXEL_PER_CELL = 50

# --- Normalization Constants ---
# These values are used to scale player stats to a [0, 1] range.
MAX_BOMB_PACK = 5.0
MAX_POTION = 5.0
MAX_BOOTS = 3.0  # 用户确认：最多只能有3个鞋子
# Max current bombs is 1 (initial) + max upgrades
MAX_CURRENT_BOMBS = MAX_BOMB_PACK
# New constants for real effects
BASE_SPEED = 10.0
SPEED_PER_BOOT = 2.0
MAX_SPEED = (BASE_SPEED + MAX_BOOTS * SPEED_PER_BOOT) * 2.0 # *2 for acceleration point
RANGE_PER_POTION = 1.0
MAX_RANGE = MAX_POTION * RANGE_PER_POTION

# --- Action Space Configuration ---
# 新的动作空间：MultiDiscrete([5, 2, 5])
# 
# 第1维 - 方向 (5个选项):
#   0: 不动 (Stay)
#   1: 上 (Up)
#   2: 下 (Down)
#   3: 左 (Left)
#   4: 右 (Right)
#
# 第2维 - 放炸弹 (2个选项):
#   0: 不放炸弹
#   1: 放炸弹
#
# 第3维 - 速度档位 (5个选项) - 相对于当前最大速度的百分比:
#   游戏机制说明：
#     - 基础速度：10像素/tick
#     - 每个鞋子：+2像素/tick（最多3个）
#     - 最大速度：10 + 3*2 = 16像素/tick
#     - 踩加速点：速度×2 = 32像素/tick
#     - stride=0时为最大速度（动态）
#
#   速度档位（相对百分比）:
#   0: 最大速度 (100%) - stride=0 - 快速移动
#   1: 极慢速 (20%) - 用于精确定位，通过狭窄区域
#   2: 慢速 (40%) - 用于精确控制
#   3: 中速 (60%) - 用于平衡速度和控制
#   4: 快速 (80%) - 较快但仍可控
#
# 实际速度计算：
#   - 先计算当前最大速度 = BASE_SPEED + boots_count * SPEED_PER_BOOT
#   - 如果踩加速点，最大速度 ×2
#   - 根据档位计算实际stride：
#       档位1-4: stride = int(max_speed * 百分比)
#       档位0: stride = 0（表示最大速度）
#
# 示例（3个鞋子，未踩加速点，max_speed=16）:
#   档位0: stride=0 (16像素/tick)
#   档位1: stride=3 (3像素/tick, 20%)
#   档位2: stride=6 (6像素/tick, 40%)
#   档位3: stride=10 (10像素/tick, 60%)
#   档位4: stride=13 (13像素/tick, 80%)
#
# 示例动作: [1, 1, 0] = 向上移动 + 放炸弹 + 最大速度

# 速度档位到百分比的映射
SPEED_GEAR_PERCENTAGES = {
    0: 1.0,   # 100% - 最大速度（stride=0）
    1: 0.2,   # 20% - 极慢
    2: 0.4,   # 40% - 慢
    3: 0.6,   # 60% - 中
    4: 0.8,   # 80% - 快
}
