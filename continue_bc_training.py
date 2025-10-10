"""
继续训练已有的BC模型
"""
import torch
import numpy as np
from train_bc_weighted import train_bc_weighted, load_expert_trajectories, WeightedBCTrainer, analyze_action_distribution
from environment import BomberEnv
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import PPO
import os


def continue_bc_training(
    checkpoint_path: str,
    data_dir: str = "./expert_data",
    output_dir: str = "./bc_models_weighted",
    additional_epochs: int = 100,
    batch_size: int = 128,
    learning_rate: float = 3e-4,
    use_weights: bool = True,
):
    """
    继续训练已有的BC模型
    
    Args:
        checkpoint_path: 已训练模型路径 (.pt 或 .pth)
        data_dir: 专家数据目录
        output_dir: 输出目录
        additional_epochs: 额外训练的轮数
        batch_size: 批大小
        learning_rate: 学习率
        use_weights: 是否使用加权
    """
    print("="*60)
    print("继续BC训练")
    print("="*60)
    print(f"\n配置:")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  数据目录: {data_dir}")
    print(f"  输出目录: {output_dir}")
    print(f"  额外训练: {additional_epochs} epochs")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  使用加权: {use_weights}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  设备: {device}\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 分析动作分布
    action_weights = None
    if use_weights:
        action_stats = analyze_action_distribution(data_dir)
        action_weights = {
            'direction': action_stats['direction'],
            'bomb': action_stats['bomb'],
            'speed': action_stats['speed']
        }
    
    # 2. 加载数据
    trajectories = load_expert_trajectories(data_dir)
    
    # 3. 创建环境
    base_env = BomberEnv()
    env = FlattenObservation(base_env)
    
    print(f"环境信息:")
    print(f"  Observation Space: {env.observation_space}")
    print(f"  Action Space: {env.action_space}\n")
    
    # 4. 加载已训练的Policy
    print(f"{'='*60}")
    print("加载已训练的Policy...")
    print(f"{'='*60}\n")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"模型文件不存在: {checkpoint_path}")
    
    # 创建policy结构
    rng = np.random.default_rng(seed=42)
    
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64])
    )
    
    policy = ActorCriticPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=lambda _: learning_rate,
        **policy_kwargs
    ).to(device)
    
    # 加载权重
    if checkpoint_path.endswith('.pt'):
        # 完整对象
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        policy.load_state_dict(checkpoint.state_dict())
        print(f"✅ 从完整对象加载: {checkpoint_path}")
    else:
        # state_dict
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
        policy.load_state_dict(state_dict)
        print(f"✅ 从state_dict加载: {checkpoint_path}")
    
    print(f"✅ Policy已加载 (net_arch=[256, 128, 64])\n")
    
    # 5. 创建BC训练器
    if use_weights:
        bc_trainer = WeightedBCTrainer(
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=trajectories,
            policy=policy,
            rng=rng,
            batch_size=batch_size,
            ent_weight=1e-3,
            l2_weight=5e-5,
            device=device,
            action_weights=action_weights,
        )
        print(f"✅ 加权BC训练器已创建\n")
    else:
        from imitation.algorithms import bc
        bc_trainer = bc.BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=trajectories,
            policy=policy,
            rng=rng,
            batch_size=batch_size,
            ent_weight=1e-3,
            l2_weight=1e-4,
            device=device,
        )
        print(f"✅ 标准BC训练器已创建\n")
    
    # 6. 继续训练
    print(f"{'='*60}")
    print(f"继续训练 {additional_epochs} epochs...")
    print(f"{'='*60}\n")
    
    try:
        bc_trainer.train(n_epochs=additional_epochs)
        
        print(f"\n{'='*60}")
        print("✅ 训练完成！")
        print(f"{'='*60}\n")
        
    except KeyboardInterrupt:
        print(f"\n{'='*60}")
        print("⚠️  训练被用户中断")
        print(f"{'='*60}\n")
    
    # 7. 保存模型（添加版本号）
    policy_path = f"{output_dir}/bc_policy_weighted_continued.pth"
    torch.save(bc_trainer.policy.state_dict(), policy_path)
    print(f"✅ Policy已保存: {policy_path}")
    
    policy_obj_path = f"{output_dir}/bc_policy_weighted_continued.pt"
    torch.save(bc_trainer.policy, policy_obj_path)
    print(f"✅ Policy对象已保存: {policy_obj_path}")
    
    # 8. 转换为PPO格式
    print(f"\n{'='*60}")
    print("转换为PPO格式...")
    print(f"{'='*60}\n")
    
    try:
        ppo_policy_kwargs = dict(
            net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64])
        )
        
        ppo_model = PPO(
            policy=ActorCriticPolicy,
            env=env,
            policy_kwargs=ppo_policy_kwargs,
            learning_rate=learning_rate,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            device=device,
        )
        
        ppo_model.policy.load_state_dict(bc_trainer.policy.state_dict(), strict=False)
        
        ppo_path = f"{output_dir}/bc_ppo_weighted_continued.zip"
        ppo_model.save(ppo_path)
        print(f"✅ PPO格式已保存: {ppo_path}")
        
    except Exception as e:
        print(f"⚠️  PPO转换失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 9. 总结
    print(f"\n{'='*60}")
    print("继续训练完成总结")
    print(f"{'='*60}")
    print(f"✅ 新Policy: {policy_obj_path}")
    if 'ppo_path' in locals():
        print(f"✅ 新PPO格式: {ppo_path}")
    print(f"\n下一步:")
    print(f"   1. 测试新模型效果")
    print(f"   2. 如果满意，替换旧模型:")
    print(f"      mv {policy_obj_path} {output_dir}/bc_policy_weighted.pt")
    print(f"      mv {ppo_path} {output_dir}/bc_ppo_weighted.zip")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="继续训练BC模型")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="已训练模型路径 (.pt 或 .pth)"
    )
    parser.add_argument("--data_dir", type=str, default="./expert_data")
    parser.add_argument("--output_dir", type=str, default="./bc_models_weighted")
    parser.add_argument("--epochs", type=int, default=100, help="额外训练的轮数")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--no_weights", action="store_true", help="禁用加权损失")
    
    args = parser.parse_args()
    
    continue_bc_training(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        additional_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_weights=not args.no_weights,
    )

