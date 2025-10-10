"""
测试训练好的模型，诊断问题
"""
import numpy as np
from stable_baselines3 import PPO
from environment import BomberEnv
from gymnasium.wrappers import FlattenObservation
import os


def test_model_predictions(model_path: str, n_tests: int = 100):
    """
    测试模型预测，看是否真的在学习
    """
    print("="*60)
    print("模型预测测试")
    print("="*60)
    print(f"模型路径: {model_path}")
    print(f"测试次数: {n_tests}\n")
    
    # 检查文件
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    # 加载模型
    print("加载模型...")
    try:
        base_env = BomberEnv()
        env = FlattenObservation(base_env)
        model = PPO.load(model_path, env=env)
        print(f"✅ 模型加载成功\n")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 重置环境
    obs, info = env.reset()
    
    # 统计动作分布
    direction_counts = np.zeros(5, dtype=int)  # 0-4
    bomb_counts = np.zeros(2, dtype=int)       # 0-1
    speed_counts = np.zeros(5, dtype=int)      # 0-4
    
    print("="*60)
    print("开始预测测试...")
    print("="*60)
    
    for i in range(n_tests):
        # 模型预测
        action, _states = model.predict(obs, deterministic=False)
        
        # 统计
        direction_counts[action[0]] += 1
        bomb_counts[action[1]] += 1
        speed_counts[action[2]] += 1
        
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i < 10:  # 打印前10次预测
            print(f"预测 {i+1}: 方向={action[0]}, 炸弹={action[1]}, 速度={action[2]}")
        
        if terminated or truncated:
            obs, info = env.reset()
    
    # 分析结果
    print(f"\n{'='*60}")
    print("预测结果分析")
    print(f"{'='*60}\n")
    
    direction_labels = ['停止', '上', '下', '左', '右']
    print("【方向分布】")
    for i, label in enumerate(direction_labels):
        pct = direction_counts[i] / n_tests * 100
        bar = "█" * int(pct / 2)
        print(f"  {label:4s}: {direction_counts[i]:3d} ({pct:5.1f}%) {bar}")
    
    print("\n【炸弹分布】")
    bomb_labels = ['不放', '放炸弹']
    for i, label in enumerate(bomb_labels):
        pct = bomb_counts[i] / n_tests * 100
        bar = "█" * int(pct / 2)
        marker = " ⚠️" if (i == 1 and pct < 1.0) else ""
        print(f"  {label:6s}: {bomb_counts[i]:3d} ({pct:5.1f}%) {bar}{marker}")
    
    print("\n【速度分布】")
    speed_labels = ['最快(100%)', '极慢(20%)', '慢(40%)', '中(60%)', '快(80%)']
    for i, label in enumerate(speed_labels):
        pct = speed_counts[i] / n_tests * 100
        bar = "█" * int(pct / 2)
        marker = " ⚠️" if (i > 0 and pct < 1.0) else ""
        print(f"  {label:10s}: {speed_counts[i]:3d} ({pct:5.1f}%) {bar}{marker}")
    
    # 诊断
    print(f"\n{'='*60}")
    print("诊断结果")
    print(f"{'='*60}\n")
    
    issues = []
    
    # 检查是否所有动作都是相同的（没有学习）
    if direction_counts.max() == n_tests:
        issues.append("❌ 方向动作始终相同（模型未学习）")
    
    if bomb_counts[1] == 0:
        issues.append("❌ 从不放炸弹（关键功能缺失）")
    elif bomb_counts[1] < n_tests * 0.01:
        issues.append(f"⚠️  放炸弹频率过低 ({bomb_counts[1]/n_tests*100:.1f}%)")
    
    if speed_counts[0] == n_tests:
        issues.append("⚠️  速度始终为最快（未学习速度控制）")
    
    if len(issues) == 0:
        print("✅ 模型看起来在正常工作")
        print("   - 动作有多样性")
        print("   - 会放炸弹")
        print("   - 有速度变化")
    else:
        print("检测到以下问题:")
        for issue in issues:
            print(f"  {issue}")
        
        print("\n可能的原因:")
        print("  1. 训练轮数不够（prob_true_act 太低）")
        print("  2. 网络容量不足（考虑增大到 [512, 256, 128]）")
        print("  3. 观察空间处理有问题")
        print("  4. 学习率过高或过低")
        print("  5. 专家数据质量问题")
    
    print(f"\n{'='*60}\n")


def compare_with_random(model_path: str, n_tests: int = 100):
    """
    对比模型和随机AI的行为
    """
    print("="*60)
    print("模型 vs 随机AI对比")
    print("="*60)
    
    # 加载模型
    base_env = BomberEnv()
    env = FlattenObservation(base_env)
    model = PPO.load(model_path, env=env)
    
    # 测试模型
    obs, info = env.reset()
    model_actions = []
    for _ in range(n_tests):
        action, _ = model.predict(obs, deterministic=False)
        model_actions.append(action)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    
    # 测试随机
    obs, info = env.reset()
    random_actions = []
    for _ in range(n_tests):
        action = env.action_space.sample()
        random_actions.append(action)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    
    # 计算熵（多样性）
    model_actions = np.array(model_actions)
    random_actions = np.array(random_actions)
    
    model_direction_entropy = -np.sum(
        [np.sum(model_actions[:, 0] == i) / n_tests * 
         np.log(np.sum(model_actions[:, 0] == i) / n_tests + 1e-10) 
         for i in range(5)]
    )
    random_direction_entropy = np.log(5)  # 均匀分布
    
    print(f"\n方向选择熵:")
    print(f"  模型:   {model_direction_entropy:.3f}")
    print(f"  随机:   {random_direction_entropy:.3f} (理论最大)")
    print(f"  相似度: {(model_direction_entropy / random_direction_entropy * 100):.1f}%")
    
    if model_direction_entropy / random_direction_entropy > 0.9:
        print("\n⚠️  模型行为接近随机！可能训练失败。")
    else:
        print("\n✅ 模型有明确的策略偏好。")
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试模型")
    parser.add_argument(
        "--model",
        type=str,
        default="./bc_models_weighted/bc_ppo_weighted.zip",
        help="模型路径"
    )
    parser.add_argument(
        "--n_tests",
        type=int,
        default=100,
        help="测试次数"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="对比随机AI"
    )
    
    args = parser.parse_args()
    
    # 测试模型预测
    test_model_predictions(args.model, args.n_tests)
    
    # 对比随机AI
    if args.compare:
        compare_with_random(args.model, args.n_tests)

