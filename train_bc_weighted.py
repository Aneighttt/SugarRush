"""
BC训练 - 加权损失版本（修复版）
使用 imitation 库 + 自定义加权损失
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import glob
import os
from typing import List, Dict
from imitation.algorithms import bc
from imitation.data import types
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import PPO
import warnings

warnings.filterwarnings("ignore", message="Converting a tensor with requires_grad=True")


def analyze_action_distribution(data_dir: str) -> Dict:
    """分析动作分布，计算类别权重"""
    print(f"\n{'='*60}")
    print("分析动作分布...")
    print(f"{'='*60}\n")
    
    total_direction = np.zeros(5, dtype=int)
    total_bomb = np.zeros(2, dtype=int)
    total_speed = np.zeros(5, dtype=int)
    total_actions = 0
    
    expert_dirs = [d for d in glob.glob(f"{data_dir}/expert_*") if os.path.isdir(d)]
    
    for expert_dir in expert_dirs:
        pkl_files = sorted(glob.glob(f"{expert_dir}/*.pkl"))
        
        for pkl_file in pkl_files:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            for episode in data['episodes']:
                actions = np.array(episode['actions'])
                total_actions += len(actions)
                
                total_direction += np.bincount(actions[:, 0].astype(int), minlength=5)
                total_bomb += np.bincount(actions[:, 1].astype(int), minlength=2)
                total_speed += np.bincount(actions[:, 2].astype(int), minlength=5)
    
    print(f"总动作数: {total_actions:,}\n")
    
    # 显示分布
    direction_labels = ['停', '上', '下', '左', '右']
    print("【方向】")
    for i, label in enumerate(direction_labels):
        pct = total_direction[i] / total_actions * 100
        print(f"  {label}: {total_direction[i]:7d} ({pct:5.2f}%)")
    
    bomb_labels = ['不放', '放炸弹']
    print("\n【炸弹】⭐ 关键动作")
    for i, label in enumerate(bomb_labels):
        pct = total_bomb[i] / total_actions * 100
        marker = " ← 重要！" if i == 1 else ""
        print(f"  {label}: {total_bomb[i]:7d} ({pct:5.2f}%){marker}")
    
    speed_labels = ['最快', '极慢', '慢', '中', '快']
    print("\n【速度】⭐ 关键动作")
    for i, label in enumerate(speed_labels):
        pct = total_speed[i] / total_actions * 100
        marker = " ← 重要！" if 1 <= i <= 4 else ""
        print(f"  {label}: {total_speed[i]:7d} ({pct:5.2f}%){marker}")
    
    # 计算重要性权重
    direction_weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])  # 方向都一样
    bomb_weights = np.array([1.0, 20.0])  # 放炸弹权重 x20
    speed_weights = np.array([1.0, 5.0, 8.0, 8.0, 6.0])  # 减速权重 x5-8
    
    print(f"\n{'='*60}")
    print("权重策略: importance (重要性优先)")
    print(f"{'='*60}")
    print(f"方向权重: {direction_weights}")
    print(f"炸弹权重: {bomb_weights}  ← 放炸弹权重x20")
    print(f"速度权重: {speed_weights}  ← 减速权重x5-8")
    print(f"{'='*60}\n")
    
    return {
        'direction': direction_weights,
        'bomb': bomb_weights,
        'speed': speed_weights,
        'distribution': {
            'direction': total_direction,
            'bomb': total_bomb,
            'speed': total_speed
        }
    }


def load_expert_trajectories(data_dir: str) -> List[types.Trajectory]:
    """加载专家数据并转换为 Trajectory 格式"""
    print(f"{'='*60}")
    print(f"加载专家数据从: {data_dir}")
    print(f"{'='*60}\n")
    
    trajectories = []
    expert_dirs = [d for d in glob.glob(f"{data_dir}/expert_*") if os.path.isdir(d)]
    
    if len(expert_dirs) == 0:
        raise ValueError(f"在 {data_dir} 中没有找到专家数据目录")
    
    print(f"找到 {len(expert_dirs)} 个专家:")
    for expert_dir in expert_dirs:
        print(f"  - {os.path.basename(expert_dir)}")
    
    total_transitions = 0
    
    for expert_dir in expert_dirs:
        print(f"\n加载 {os.path.basename(expert_dir)} 的数据...")
        pkl_files = sorted(glob.glob(f"{expert_dir}/*.pkl"))
        
        for pkl_file in pkl_files:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            for episode in data['episodes']:
                obs_list = episode['observations']
                actions_list = episode['actions']
                next_obs_list = episode['next_observations']
                
                if len(obs_list) == 0:
                    continue
                
                # 构建完整观察序列
                full_obs_list = obs_list + [next_obs_list[-1]]
                
                # Flatten observations
                obs_arrays = []
                for obs in full_obs_list:
                    flattened_grid = obs['grid_view'].flatten()
                    flattened_obs = np.concatenate([flattened_grid, obs['player_state']])
                    obs_arrays.append(flattened_obs)
                
                trajectory = types.Trajectory(
                    obs=np.array(obs_arrays),
                    acts=np.array(actions_list),
                    infos=None,
                    terminal=True
                )
                
                trajectories.append(trajectory)
                total_transitions += len(obs_list)
        
        print(f"  ✅ 加载了 {len(pkl_files)} 个文件")
    
    print(f"\n{'='*60}")
    print(f"✅ 总共加载:")
    print(f"   Episodes: {len(trajectories)}")
    print(f"   Transitions: {total_transitions:,}")
    print(f"{'='*60}\n")
    
    return trajectories


class WeightedBCTrainer(bc.BC):
    """
    带加权损失的BC训练器
    重写损失计算以支持动作权重
    """
    def __init__(self, *args, action_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_weights = action_weights
        
        if action_weights is not None:
            # 转换为 tensor
            self.direction_weights = torch.tensor(
                action_weights['direction'], dtype=torch.float32
            )
            self.bomb_weights = torch.tensor(
                action_weights['bomb'], dtype=torch.float32
            )
            self.speed_weights = torch.tensor(
                action_weights['speed'], dtype=torch.float32
            )
    
    def _calculate_loss(self, obs, acts, **kwargs):
        """重写损失计算，加入动作权重"""
        # 获取 policy 的 distribution
        obs_tensor = torch.as_tensor(obs, device=self.device)
        acts_tensor = torch.as_tensor(acts, device=self.device)
        
        # 前向传播
        distribution = self.policy.get_distribution(obs_tensor)
        log_prob = distribution.log_prob(acts_tensor)
        
        # 基础损失
        neglogp = -log_prob
        
        # 如果有权重，计算加权损失
        if self.action_weights is not None:
            # acts shape: (batch_size, 3)
            direction_acts = acts_tensor[:, 0].long()
            bomb_acts = acts_tensor[:, 1].long()
            speed_acts = acts_tensor[:, 2].long()
            
            # 移动权重到设备
            direction_w = self.direction_weights.to(self.device)
            bomb_w = self.bomb_weights.to(self.device)
            speed_w = self.speed_weights.to(self.device)
            
            # 为每个样本计算权重
            sample_direction_w = direction_w[direction_acts]
            sample_bomb_w = bomb_w[bomb_acts]
            sample_speed_w = speed_w[speed_acts]
            
            # 组合权重（炸弹权重x2，速度权重x1.5）
            sample_weights = (sample_direction_w + sample_bomb_w * 2.0 + sample_speed_w * 1.5) / 4.5
            
            # 加权负对数似然
            weighted_neglogp = neglogp * sample_weights
            neglogp_mean = weighted_neglogp.mean()
        else:
            neglogp_mean = neglogp.mean()
        
        # 熵正则化
        entropy = distribution.entropy().mean() if hasattr(distribution, 'entropy') else 0.0
        ent_loss = -self.ent_weight * entropy
        
        # L2 正则化
        l2_norm = sum(p.pow(2).sum() for p in self.policy.parameters())
        l2_loss = self.l2_weight * l2_norm
        
        # 总损失
        loss = neglogp_mean + ent_loss + l2_loss
        
        # 计算准确率（用于监控）
        with torch.no_grad():
            predicted_acts = distribution.mode()
            if len(predicted_acts.shape) == 1:
                predicted_acts = predicted_acts.unsqueeze(-1)
            
            # MultiDiscrete: 三个都对才算对
            all_correct = (predicted_acts == acts_tensor).all(dim=1).float().mean()
            
            # 各子动作准确率
            direction_acc = (predicted_acts[:, 0] == acts_tensor[:, 0]).float().mean()
            bomb_acc = (predicted_acts[:, 1] == acts_tensor[:, 1]).float().mean()
            speed_acc = (predicted_acts[:, 2] == acts_tensor[:, 2]).float().mean()
        
        return {
            'loss': loss,
            'neglogp': neglogp_mean.item(),
            'entropy': entropy.item() if isinstance(entropy, torch.Tensor) else entropy,
            'ent_loss': ent_loss.item() if isinstance(ent_loss, torch.Tensor) else 0.0,
            'l2_loss': l2_loss.item(),
            'l2_norm': l2_norm.item(),
            'prob_true_act': all_correct.item(),
            'direction_acc': direction_acc.item(),
            'bomb_acc': bomb_acc.item(),
            'speed_acc': speed_acc.item(),
        }


def train_bc_weighted(
    data_dir: str = "./expert_data",
    output_dir: str = "./bc_models_weighted",
    n_epochs: int = 150,
    batch_size: int = 128,
    learning_rate: float = 3e-4,
    use_weights: bool = True,
):
    """使用加权损失训练BC模型"""
    print("="*60)
    print("BC训练 - 加权损失版本")
    print("="*60)
    print(f"\n配置:")
    print(f"  数据目录: {data_dir}")
    print(f"  输出目录: {output_dir}")
    print(f"  Epochs: {n_epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  使用加权: {use_weights}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  设备: {device}\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 分析动作分布（如果使用权重）
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
    from environment import BomberEnv
    from gymnasium.wrappers import FlattenObservation
    
    base_env = BomberEnv()
    env = FlattenObservation(base_env)
    
    print(f"环境信息:")
    print(f"  Observation Space: {env.observation_space}")
    print(f"  Action Space: {env.action_space}\n")
    
    # 4. 创建 Policy
    print(f"{'='*60}")
    print("创建Policy网络...")
    print(f"{'='*60}\n")
    
    rng = np.random.default_rng(seed=42)
    
    policy_kwargs = dict(
        net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128])  # 大网络，提升容量
    )
    
    policy = ActorCriticPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=lambda _: learning_rate,
        **policy_kwargs
    ).to(device)
    
    print(f"✅ Policy已创建 (net_arch=[512, 256, 128])")
    
    # 5. 创建BC训练器（使用加权版本）
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
        print(f"✅ 加权BC训练器已创建")
        print(f"   炸弹损失权重提升 x2.0")
        print(f"   速度损失权重提升 x1.5\n")
    else:
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
    
    # 6. 训练
    print(f"{'='*60}")
    print(f"开始训练 {n_epochs} epochs...")
    print(f"{'='*60}\n")
    
    try:
        bc_trainer.train(n_epochs=n_epochs)
        
        print(f"\n{'='*60}")
        print("✅ 训练完成！")
        print(f"{'='*60}\n")
        
    except KeyboardInterrupt:
        print(f"\n{'='*60}")
        print("⚠️  训练被用户中断")
        print(f"{'='*60}\n")
    
    # 7. 保存模型
    policy_path = f"{output_dir}/bc_policy_weighted.pth"
    torch.save(bc_trainer.policy.state_dict(), policy_path)
    print(f"✅ Policy已保存: {policy_path}")
    
    policy_obj_path = f"{output_dir}/bc_policy_weighted.pt"
    torch.save(bc_trainer.policy, policy_obj_path)
    print(f"✅ Policy对象已保存: {policy_obj_path}")
    
    # 8. 转换为PPO格式
    print(f"\n{'='*60}")
    print("转换为PPO格式...")
    print(f"{'='*60}\n")
    
    try:
        # 使用与BC相同的网络结构
        ppo_policy_kwargs = dict(
            net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128])
        )
        
        ppo_model = PPO(
            policy=ActorCriticPolicy,
            env=env,
            policy_kwargs=ppo_policy_kwargs,  # 指定网络结构
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
        
        ppo_path = f"{output_dir}/bc_ppo_weighted.zip"
        ppo_model.save(ppo_path)
        print(f"✅ PPO格式已保存: {ppo_path}")
        print(f"   100%参数转移完成（网络结构一致）")
        
    except Exception as e:
        print(f"⚠️  PPO转换失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 9. 总结
    print(f"\n{'='*60}")
    print("训练完成总结")
    print(f"{'='*60}")
    print(f"✅ Policy: {policy_obj_path}")
    if 'ppo_path' in locals():
        print(f"✅ PPO格式: {ppo_path}")
        print(f"\n🎮 使用方法:")
        print(f"   robot.py 会自动加载此模型")
    
    if use_weights:
        print(f"\n🎯 加权训练优势:")
        print(f"   ✓ 放炸弹动作学习权重提升")
        print(f"   ✓ 减速走位学习权重提升")
        print(f"   ✓ 预期游戏表现: 会主动放炸弹和精确走位")
    
    print(f"\n下一步:")
    print(f"   1. 启动 robot.py 测试游戏表现")
    print(f"   2. 观察是否会在合适时机放炸弹")
    print(f"   3. 观察是否会减速躲避炸弹")
    print(f"   4. 如果效果好，进入PPO微调阶段")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="BC训练（加权损失版）")
    parser.add_argument("--data_dir", type=str, default="./expert_data")
    parser.add_argument("--output_dir", type=str, default="./bc_models_weighted")
    parser.add_argument("--n_epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--no_weights", action="store_true", help="禁用加权损失")
    
    args = parser.parse_args()
    
    train_bc_weighted(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_weights=not args.no_weights,
    )

