"""
诊断观察空间处理问题
对比专家数据收集和实际推理时的observation
"""
import numpy as np
import pickle
from pathlib import Path


def check_expert_data_observation():
    """检查专家数据中的observation"""
    print("="*60)
    print("检查专家数据的observation")
    print("="*60)
    
    # 找一个专家数据文件
    data_files = list(Path('./expert_data/expert_1').glob('expert_1_ep*.pkl'))
    if not data_files:
        print("❌ 未找到专家数据")
        return
    
    with open(data_files[0], 'rb') as f:
        data = pickle.load(f)
    
    episode = data['episodes'][0]
    obs = episode['observations'][0]
    
    print(f"\n专家数据observation结构:")
    print(f"  Keys: {obs.keys()}")
    print(f"  grid_view shape: {obs['grid_view'].shape}")
    print(f"  grid_view dtype: {obs['grid_view'].dtype}")
    print(f"  grid_view range: [{obs['grid_view'].min():.3f}, {obs['grid_view'].max():.3f}]")
    print(f"  grid_view mean: {obs['grid_view'].mean():.3f}")
    print(f"  grid_view 非零比例: {(obs['grid_view'] != 0).sum() / obs['grid_view'].size * 100:.1f}%")
    
    print(f"\n  player_state shape: {obs['player_state'].shape}")
    print(f"  player_state dtype: {obs['player_state'].dtype}")
    print(f"  player_state: {obs['player_state']}")
    print(f"  player_state range: [{obs['player_state'].min():.3f}, {obs['player_state'].max():.3f}]")
    
    # 检查grid_view各通道
    print(f"\n  grid_view各通道统计:")
    for i in range(obs['grid_view'].shape[0]):
        channel = obs['grid_view'][i]
        non_zero = (channel != 0).sum()
        print(f"    通道{i:2d}: 非零={non_zero:4d} ({non_zero/channel.size*100:5.2f}%), "
              f"min={channel.min():.3f}, max={channel.max():.3f}")
    
    return obs


def check_frame_processor():
    """检查frame_processor的处理"""
    print(f"\n{'='*60}")
    print("检查frame_processor逻辑")
    print("="*60)
    
    # 读取frame_processor代码
    with open('frame_processor.py', 'r') as f:
        content = f.read()
    
    # 检查关键函数
    checks = {
        'create_grid_view归一化': 'def create_grid_view' in content,
        'player_state归一化': 'def create_player_state' in content,
        'MAX_BOMBS定义': 'MAX_BOMBS' in content,
        'MAX_RANGE定义': 'MAX_RANGE' in content,
        'grid归一化': '/ 1.0' in content or '.astype(np.float32)' in content,
    }
    
    print("\n代码检查:")
    for check, result in checks.items():
        status = "✅" if result else "❌"
        print(f"  {status} {check}")
    
    # 查找归一化相关代码
    print("\n关键归一化代码片段:")
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'normalize' in line.lower() or '/ ' in line and 'float' in line:
            print(f"  Line {i+1}: {line.strip()}")


def simulate_observation_processing():
    """模拟observation处理流程"""
    print(f"\n{'='*60}")
    print("模拟observation处理")
    print("="*60)
    
    # 检查专家数据
    obs = check_expert_data_observation()
    
    if obs is None:
        return
    
    # 模拟flatten
    flattened_grid = obs['grid_view'].flatten()
    flattened_obs = np.concatenate([flattened_grid, obs['player_state']])
    
    print(f"\nFlatten后:")
    print(f"  Total size: {flattened_obs.shape[0]}")
    print(f"  Expected: 6282 (14*16*28 + 10)")
    print(f"  Range: [{flattened_obs.min():.3f}, {flattened_obs.max():.3f}]")
    print(f"  Mean: {flattened_obs.mean():.3f}")
    print(f"  Std: {flattened_obs.std():.3f}")
    
    # 检查是否有异常值
    if flattened_obs.max() > 10:
        print(f"  ⚠️  有异常大的值！可能归一化有问题")
    
    if flattened_obs.min() < -1:
        print(f"  ⚠️  有异常小的值！可能归一化有问题")
    
    if (flattened_obs == 0).sum() / flattened_obs.size > 0.95:
        print(f"  ⚠️  超过95%的值为0，可能observation信息不足")


def check_realtime_collector():
    """检查实时收集器的observation处理"""
    print(f"\n{'='*60}")
    print("检查实时数据收集器")
    print("="*60)
    
    with open('realtime_expert_collector.py', 'r') as f:
        content = f.read()
    
    # 检查是否使用了相同的frame_processor
    if 'from frame_processor import' in content:
        print("✅ 使用frame_processor处理observation")
    else:
        print("❌ 未使用frame_processor，可能有不一致！")
    
    # 检查grid_view创建
    if 'create_grid_view_for_player' in content:
        print("✅ 为每个玩家创建grid_view")
    else:
        print("⚠️  grid_view创建方式可能不同")


if __name__ == "__main__":
    print("\n")
    print("*" * 60)
    print(" 观察空间一致性诊断")
    print("*" * 60)
    
    # 1. 检查专家数据
    check_expert_data_observation()
    
    # 2. 检查frame_processor
    check_frame_processor()
    
    # 3. 模拟处理
    simulate_observation_processing()
    
    # 4. 检查实时收集器
    check_realtime_collector()
    
    print(f"\n{'='*60}")
    print("诊断完成")
    print("="*60)
    print("\n如果发现问题，需要:")
    print("  1. 确保训练数据和推理使用相同的observation处理")
    print("  2. 确保所有值都正确归一化到[0,1]或[-1,1]")
    print("  3. 确保grid_view和player_state的维度一致")
    print("  4. 重新收集数据或重新训练")
    print("="*60)

