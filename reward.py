from data_models import Frame

def calculate_reward(frame: Frame, prev_frame: Frame) -> float:
    """
    根据当前帧和前一帧计算奖励。

    这是一个独立的奖励模块接口，您可以根据需求修改此函数的内部逻辑。

    当前奖励设计:
    - 摧毁软墙: +0.1
    - 拾取道具: +0.5
    - 消灭敌人: +1.0
    - 我方死亡: -1.0
    - 生存时间: +0.001 (每 tick)
    """
    reward = 0.001  # 生存奖励

    if not prev_frame:
        return reward

    # 1. 比较软墙数量
    current_soft_walls = sum(1 for row in frame.map for cell in row if cell.terrain == 'soft_wall')
    prev_soft_walls = sum(1 for row in prev_frame.map for cell in row if cell.terrain == 'soft_wall')
    if current_soft_walls < prev_soft_walls:
        reward += 0.1 * (prev_soft_walls - current_soft_walls)

    # 2. 比较道具数量
    if len(frame.map_items) < len(prev_frame.map_items):
        reward += 0.5

    # 3. 比较敌方玩家数量
    current_alive_enemies = sum(1 for p in frame.other_players if p.status == 'alive')
    prev_alive_enemies = sum(1 for p in prev_frame.other_players if p.status == 'alive')
    if current_alive_enemies < prev_alive_enemies:
        reward += 1.0 * (prev_alive_enemies - current_alive_enemies)

    # 4. 检查我方玩家是否死亡
    if frame.my_player.status == 'dead' and prev_frame.my_player.status == 'alive':
        reward -= 1.0

    return reward
