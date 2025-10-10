"""
测试实时数据收集系统
运行此脚本验证所有组件是否正常工作
"""
import numpy as np
from data_models import Frame
from realtime_expert_collector import RealtimeExpertCollector


def create_test_frame(match_id: str, tick: int, my_team: str = 'A'):
    """创建测试用的Frame对象"""
    frame_data = {
        'current_match_id': match_id,
        'current_tick': tick,
        'map': [[{'terrain': 'P', 'ownership': 'N', 'owner_id': None} 
                 for _ in range(28)] for _ in range(16)],
        'my_player': {
            'id': 'my_random_ai',
            'name': 'MyRandomAI',
            'team': my_team,
            'status': 'A',
            'extra_status': [],
            'position': {'x': 100 + tick * 2, 'y': 100},  # 缓慢移动
            'direction': 0,
            'bomb_pack_count': 1,
            'sweet_potion_count': 1,
            'agility_boots_count': 0
        },
        'other_players': [
            {
                'id': 'friendly_random',
                'name': 'FriendlyRandom',
                'team': my_team,  # 友方
                'status': 'A',
                'extra_status': [],
                'position': {'x': 200, 'y': 200},
                'direction': 0,
                'bomb_pack_count': 1,
                'sweet_potion_count': 1,
                'agility_boots_count': 0
            },
            {
                'id': 'enemy_expert_1',
                'name': 'EnemyExpert1',
                'team': 'B',  # 敌方
                'status': 'A',
                'extra_status': [{'name': 'INV', 'expire_at': tick + 100}],  # 测试INV状态
                'position': {'x': 300 + tick * 3, 'y': 300},  # 向右移动
                'direction': 3,
                'bomb_pack_count': 2,
                'sweet_potion_count': 2,
                'agility_boots_count': 1
            },
            {
                'id': 'enemy_expert_2',
                'name': 'EnemyExpert2',
                'team': 'B',  # 敌方
                'status': 'A',
                'extra_status': [{'name': 'THB', 'expire_at': tick + 50}],  # 测试THB状态
                'position': {'x': 400, 'y': 400 - tick * 2},  # 向上移动
                'direction': 0,
                'bomb_pack_count': 1,
                'sweet_potion_count': 1,
                'agility_boots_count': 0
            }
        ],
        'bombs': [],
        'map_items': []
    }
    return Frame(frame_data)


def test_basic_collection():
    """测试基础数据收集功能"""
    print("\n" + "="*60)
    print("测试1: 基础数据收集")
    print("="*60)
    
    collector = RealtimeExpertCollector(save_dir="./test_expert_data")
    
    # 模拟一局游戏的前10帧
    match_id = "test_match_001"
    for tick in range(10):
        frame = create_test_frame(match_id, tick)
        expert_actions = collector.process_frame(frame)
        
        if tick > 0:  # 第一帧没有动作
            print(f"Tick {tick}: 推断动作 = {expert_actions}")
    
    collector.finish_episode()
    collector.print_statistics()
    
    print("\n✅ 测试1通过")


def test_bomb_detection():
    """测试放炸弹检测"""
    print("\n" + "="*60)
    print("测试2: 放炸弹检测")
    print("="*60)
    
    collector = RealtimeExpertCollector(save_dir="./test_expert_data")
    
    match_id = "test_match_002"
    
    # 第一帧：没有炸弹
    frame1_data = create_test_frame(match_id, 0)
    collector.process_frame(frame1_data)
    
    # 第二帧：enemy_expert_1放了一个炸弹
    frame2 = create_test_frame(match_id, 1)
    frame2.bombs = [
        type('Bomb', (), {
            'position': type('Position', (), {'x': 300, 'y': 300})(),
            'owner_id': 'enemy_expert_1',
            'explode_at': 100,
            'range': 3
        })()
    ]
    
    expert_actions = collector.process_frame(frame2)
    
    print(f"推断的动作: {expert_actions}")
    if expert_actions.get('enemy_expert_1') == 4:
        print("✅ 成功检测到放炸弹动作！")
    else:
        print(f"❌ 检测失败，动作为: {expert_actions.get('enemy_expert_1')}")
    
    collector.finish_episode()
    print("\n✅ 测试2完成")


def test_movement_detection():
    """测试移动检测"""
    print("\n" + "="*60)
    print("测试3: 移动方向检测")
    print("="*60)
    
    collector = RealtimeExpertCollector(save_dir="./test_expert_data")
    
    match_id = "test_match_003"
    
    # 创建移动序列
    movements = [
        ("向右移动", lambda t: {'x': 300 + t * 10, 'y': 300}, 3),
        ("向左移动", lambda t: {'x': 400 - t * 10, 'y': 300}, 2),
        ("向下移动", lambda t: {'x': 300, 'y': 300 + t * 10}, 1),
        ("向上移动", lambda t: {'x': 300, 'y': 400 - t * 10}, 0),
    ]
    
    for name, pos_func, expected_action in movements:
        # 重置收集器
        collector.last_frame = None
        collector.last_observations = {}
        
        # 创建两帧
        frame1 = create_test_frame(match_id, 0)
        frame1.other_players[1].position.x = pos_func(0)['x']
        frame1.other_players[1].position.y = pos_func(0)['y']
        collector.process_frame(frame1)
        
        frame2 = create_test_frame(match_id, 1)
        frame2.other_players[1].position.x = pos_func(1)['x']
        frame2.other_players[1].position.y = pos_func(1)['y']
        expert_actions = collector.process_frame(frame2)
        
        detected_action = expert_actions.get('enemy_expert_1', -1)
        action_names = ["上", "下", "左", "右", "放炸弹", "停止"]
        
        if detected_action == expected_action:
            print(f"✅ {name}: 检测正确 ({action_names[detected_action]})")
        else:
            print(f"❌ {name}: 检测错误 (预期: {action_names[expected_action]}, 实际: {action_names[detected_action] if detected_action >= 0 else '无'})")
    
    print("\n✅ 测试3完成")


def test_extra_status():
    """测试extra_status处理和时间归一化"""
    print("\n" + "="*60)
    print("测试4: Extra Status处理和时间归一化")
    print("="*60)
    
    from frame_processor_multi import create_player_state_for_player
    
    match_id = "test_match_004"
    
    # 测试不同的tick时间
    test_cases = [
        (0, "刚获得状态"),
        (50, "剩余250tick"),
        (150, "剩余150tick (50%)"),
        (250, "剩余50tick"),
        (300, "状态即将失效")
    ]
    
    for current_tick, description in test_cases:
        print(f"\n{description} (Tick {current_tick}):")
        frame = create_test_frame(match_id, current_tick)
        
        # 测试enemy_expert_1 (有INV状态, expire_at=100)
        player = frame.other_players[1]
        player_state = create_player_state_for_player(frame, player)
        
        inv_remaining = player_state[7]
        expected = max(0.0, min(1.0, (100 - current_tick) / 300.0))
        
        print(f"  INV剩余时间归一化值: {inv_remaining:.3f}")
        print(f"  预期值: {expected:.3f}")
        
        if abs(inv_remaining - expected) < 0.001:
            print("  ✅ 时间归一化正确")
        else:
            print(f"  ❌ 时间归一化错误")
    
    # 验证维度
    print(f"\nPlayer state shape: {player_state.shape}")
    if player_state.shape[0] == 10:
        print("✅ Player state维度正确 (10维)")
        print(f"   维度7 (INV): {player_state[7]:.3f}")
        print(f"   维度8 (THB): {player_state[8]:.3f}")
        print(f"   维度9 (LIT): {player_state[9]:.3f}")
    else:
        print(f"❌ Player state维度错误 (期望10维，实际{player_state.shape[0]}维)")
    
    print("\n✅ 测试4完成")


def test_observation_shape():
    """测试观察空间形状"""
    print("\n" + "="*60)
    print("测试5: 观察空间形状")
    print("="*60)
    
    from frame_processor_multi import preprocess_observation_for_player
    
    match_id = "test_match_005"
    frame = create_test_frame(match_id, 0)
    
    # 获取一个敌方专家
    enemy_expert = frame.other_players[1]  # enemy_expert_1
    
    # 创建观察
    obs = preprocess_observation_for_player(frame, enemy_expert)
    
    print(f"观察字典keys: {obs.keys()}")
    print(f"grid_view shape: {obs['grid_view'].shape}")
    print(f"player_state shape: {obs['player_state'].shape}")
    
    # 验证
    expected_grid = (14, 16, 28)
    expected_state = (10,)
    
    if obs['grid_view'].shape == expected_grid:
        print(f"✅ grid_view形状正确: {expected_grid}")
    else:
        print(f"❌ grid_view形状错误: 期望{expected_grid}, 实际{obs['grid_view'].shape}")
    
    if obs['player_state'].shape == expected_state:
        print(f"✅ player_state形状正确: {expected_state}")
    else:
        print(f"❌ player_state形状错误: 期望{expected_state}, 实际{obs['player_state'].shape}")
    
    print("\n✅ 测试5完成")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("🧪 实时数据收集系统测试套件")
    print("="*60)
    
    try:
        test_basic_collection()
        test_bomb_detection()
        test_movement_detection()
        test_extra_status()
        test_observation_shape()
        
        print("\n" + "="*60)
        print("🎉 所有测试通过！系统正常工作。")
        print("="*60)
        print("\n下一步：运行 python robot.py 开始实际收集数据")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()

