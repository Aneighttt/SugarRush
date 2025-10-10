"""
使用imitation库进行BC（行为克隆）训练
兼容Stable Baselines3，可无缝对接PPO微调
"""
import numpy as np
import torch
import pickle
import glob
import os
import warnings
from typing import List, Dict
from imitation.algorithms import bc
from imitation.data import types
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import PPO
from bc_data_collector import ExpertDataLoader

# 抑制一些已知的无害警告
warnings.filterwarnings("ignore", message="Converting a tensor with requires_grad=True")


def load_expert_trajectories(data_dir: str) -> List[types.Trajectory]:
    """
    加载专家数据并转换为imitation库的Trajectory格式
    
    Args:
        data_dir: 专家数据目录（如 "./expert_data"）
    
    Returns:
        trajectories: Trajectory对象列表
    """
    print(f"{'='*60}")
    print(f"加载专家数据从: {data_dir}")
    print(f"{'='*60}")
    
    trajectories = []
    
    # 找到所有专家目录
    expert_dirs = [d for d in glob.glob(f"{data_dir}/expert_*") if os.path.isdir(d)]
    
    if len(expert_dirs) == 0:
        raise ValueError(f"在 {data_dir} 中没有找到专家数据目录")
    
    print(f"找到 {len(expert_dirs)} 个专家:")
    for expert_dir in expert_dirs:
        print(f"  - {os.path.basename(expert_dir)}")
    
    total_transitions = 0
    
    # 遍历每个专家
    for expert_dir in expert_dirs:
        print(f"\n加载 {os.path.basename(expert_dir)} 的数据...")
        
        # 加载该专家的所有pkl文件
        pkl_files = sorted(glob.glob(f"{expert_dir}/*.pkl"))
        
        for pkl_file in pkl_files:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            # 每个pkl文件包含多个episodes
            for episode in data['episodes']:
                obs_list = episode['observations']
                actions_list = episode['actions']
                next_obs_list = episode['next_observations']
                
                if len(obs_list) == 0:
                    continue
                
                # imitation需要: len(obs) = len(actions) + 1
                # 所以我们需要 [obs_0, obs_1, ..., obs_n, next_obs_n]
                # 构建完整的观察序列：初始obs + 所有next_obs的最后一个
                full_obs_list = obs_list + [next_obs_list[-1]]
                
                # 转换观察为数组
                # 注意：imitation库需要每个observation是一个数组，不能是字典
                # 所以我们需要flatten grid_view和player_state并拼接
                obs_arrays = []
                for obs in full_obs_list:
                    # Flatten grid_view: (14, 16, 28) -> (6272,)
                    flattened_grid = obs['grid_view'].flatten()
                    # player_state: (10,)
                    # 拼接: (6272 + 10,) = (6282,)
                    flattened_obs = np.concatenate([flattened_grid, obs['player_state']])
                    obs_arrays.append(flattened_obs)
                
                # 转换为imitation的Trajectory格式
                trajectory = types.Trajectory(
                    obs=np.array(obs_arrays),  # Shape: (n+1, 6282)
                    acts=np.array(actions_list),  # Shape: (n, 3)
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


def train_bc_with_imitation(
    data_dir: str = "./expert_data",
    output_dir: str = "./bc_models_imitation",
    n_epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
):
    """
    使用imitation库训练BC模型
    
    Args:
        data_dir: 专家数据目录
        output_dir: 模型输出目录
        n_epochs: 训练轮数
        batch_size: 批大小
        learning_rate: 学习率
    """
    print("="*60)
    print("BC训练 (使用imitation库)")
    print("="*60)
    print(f"\n配置:")
    print(f"  数据目录: {data_dir}")
    print(f"  输出目录: {output_dir}")
    print(f"  Epochs: {n_epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {learning_rate}")
    
    # 检测设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  设备: {device}\n")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 加载专家轨迹
    trajectories = load_expert_trajectories(data_dir)
    
    # 2. 创建dummy环境（用于定义obs/action空间）
    from environment import BomberEnv
    from gymnasium.wrappers import FlattenObservation
    
    base_env = BomberEnv()
    env = FlattenObservation(base_env)  # Flatten Dict observation to vector
    
    print(f"环境信息:")
    print(f"  原始 Observation Space: {base_env.observation_space}")
    print(f"  Flattened Observation Space: {env.observation_space}")
    print(f"  Action Space: {env.action_space}")
    print()
    
    # 3. 创建BC训练器
    print(f"{'='*60}")
    print("创建BC训练器...")
    print(f"{'='*60}\n")
    
    # 创建随机数生成器
    rng = np.random.default_rng(seed=42)
    
    # 先创建一个自定义policy（与PPO默认配置一致）
    # PPO默认: net_arch=dict(pi=[64, 64], vf=[64, 64])
    from stable_baselines3.common.policies import ActorCriticPolicy
    
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64])  # SB3 v1.8.0+ 格式，增大容量
    )
    
    # 创建policy实例
    policy = ActorCriticPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=lambda _: 3e-4,  # 学习率schedule
        **policy_kwargs
    ).to(device)
    
    print(f"✅ Policy已创建（网络结构: [256, 128, 64]）")
    
    # 使用自定义policy创建BC训练器
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=trajectories,
        policy=policy,  # 使用自定义policy
        rng=rng,
        batch_size=batch_size,
        ent_weight=1e-3,  # 熵正则化
        l2_weight=1e-4,   # L2正则化
        device=device,
    )
    print(f"✅ BC训练器已创建")
    
    # 4. 训练
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
    
    # 5. 保存模型
    # 保存policy网络
    policy_path = f"{output_dir}/bc_policy.pth"
    torch.save(bc_trainer.policy.state_dict(), policy_path)
    print(f"✅ Policy已保存到: {policy_path}")
    
    # 保存完整的policy对象（用于直接加载）
    policy_obj_path = f"{output_dir}/bc_policy.pt"
    torch.save(bc_trainer.policy, policy_obj_path)
    print(f"✅ Policy对象已保存到: {policy_obj_path}")
    
    # 6. 转换为PPO格式（用于后续微调）
    print(f"\n{'='*60}")
    print("转换为PPO格式...")
    print(f"{'='*60}\n")
    
    try:
        # 创建PPO模型（使用与BC相同的网络结构）
        from stable_baselines3.common.policies import ActorCriticPolicy
        
        ppo_policy_kwargs = dict(
            net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64])
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
        print(f"✅ PPO模型已创建（net_arch=[256, 128, 64]）")
        
        # 加载BC策略参数到PPO
        try:
            # BC和PPO网络结构现在一致，可以完整加载
            bc_state_dict = bc_trainer.policy.state_dict()
            ppo_state_dict = ppo_model.policy.state_dict()
            
            # 检查参数兼容性
            loaded_params = 0
            skipped_params = []
            
            for name, param in bc_state_dict.items():
                if name in ppo_state_dict:
                    if param.shape == ppo_state_dict[name].shape:
                        loaded_params += 1
                    else:
                        skipped_params.append(f"{name}: BC{param.shape} vs PPO{ppo_state_dict[name].shape}")
            
            # 加载state_dict
            ppo_model.policy.load_state_dict(bc_state_dict, strict=False)
            
            print(f"✅ BC策略参数已加载到PPO")
            print(f"   加载参数: {loaded_params}/{len(bc_state_dict)}")
            
            if skipped_params:
                print(f"⚠️  跳过 {len(skipped_params)} 个不兼容参数:")
                for param_info in skipped_params[:3]:
                    print(f"     {param_info}")
                if len(skipped_params) > 3:
                    print(f"     ... 还有 {len(skipped_params)-3} 个")
            else:
                print(f"✅ 所有参数形状匹配！BC知识完整转移")
                
        except Exception as e:
            print(f"⚠️  加载BC参数时出错: {e}")
            print("   PPO将使用随机初始化")
            import traceback
            traceback.print_exc()
        
        # 保存PPO格式
        ppo_path = f"{output_dir}/bc_ppo_ready.zip"
        ppo_model.save(ppo_path)
        print(f"✅ PPO格式已保存到: {ppo_path}")
        
    except Exception as e:
        print(f"⚠️  PPO转换失败: {e}")
        print("   BC模型已保存，可以单独使用")
    
    # 7. 总结
    print(f"\n{'='*60}")
    print("训练完成总结")
    print(f"{'='*60}")
    print(f"✅ BC Policy (state_dict): {output_dir}/bc_policy.pth")
    print(f"✅ BC Policy (完整对象): {output_dir}/bc_policy.pt")
    if 'ppo_path' in locals():
        print(f"✅ PPO格式: {ppo_path}")
    print(f"\n使用方法:")
    print(f"  1. 加载BC policy:")
    print(f"     import torch")
    print(f"     policy = torch.load('{output_dir}/bc_policy.pt')")
    print(f"  2. PPO微调:")
    if 'ppo_path' in locals():
        print(f"     from stable_baselines3 import PPO")
        print(f"     model = PPO.load('{ppo_path}')")
    print(f"{'='*60}\n")


def evaluate_bc_policy(policy_path: str, n_episodes: int = 10):
    """
    评估训练好的BC策略
    
    Args:
        policy_path: 策略路径
        n_episodes: 评估轮数
    """
    from environment import BomberEnv
    
    env = BomberEnv()
    
    # 加载BC策略
    bc_policy = bc.reconstruct_policy(policy_path)
    
    print(f"评估BC策略 ({n_episodes} episodes)...")
    
    total_rewards = []
    
    for i in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = bc_policy.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
        
        total_rewards.append(episode_reward)
        print(f"  Episode {i+1}: Reward = {episode_reward:.2f}")
    
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    
    print(f"\n平均奖励: {avg_reward:.2f} ± {std_reward:.2f}")
    
    return avg_reward, std_reward


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="使用imitation库训练BC")
    parser.add_argument("--data_dir", type=str, default="./expert_data", help="专家数据目录")
    parser.add_argument("--output_dir", type=str, default="./bc_models_imitation", help="输出目录")
    parser.add_argument("--n_epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--evaluate", action="store_true", help="训练后评估模型")
    
    args = parser.parse_args()
    
    # 训练
    train_bc_with_imitation(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    
    # 评估（可选）
    if args.evaluate:
        policy_path = f"{args.output_dir}/bc_policy"
        evaluate_bc_policy(policy_path, n_episodes=10)

