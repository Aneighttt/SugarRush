"""
PPO微调训练脚本
使用BC初始化的模型进行强化学习微调
"""
import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from environment import BomberEnv
from gymnasium.wrappers import FlattenObservation


def make_env(rank, seed=0):
    """
    创建环境的工厂函数
    
    Args:
        rank: 环境编号
        seed: 随机种子
    """
    def _init():
        base_env = BomberEnv()
        env = FlattenObservation(base_env)
        env = Monitor(env)  # 记录训练统计
        env.reset(seed=seed + rank)
        return env
    return _init


def train_ppo(
    bc_model_path="./bc_models_imitation/bc_ppo_ready.zip",
    output_dir="./ppo_models",
    total_timesteps=1_000_000,
    n_envs=4,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    save_freq=50_000,
    eval_freq=25_000,
    continue_training=False,
    checkpoint_path=None,
):
    """
    PPO微调训练
    
    Args:
        bc_model_path: BC初始化的PPO模型路径
        output_dir: 输出目录
        total_timesteps: 总训练步数
        n_envs: 并行环境数量
        learning_rate: 学习率
        n_steps: 每次更新的步数
        batch_size: 批大小
        n_epochs: PPO更新的epochs
        save_freq: 保存频率
        eval_freq: 评估频率
        continue_training: 是否继续训练
        checkpoint_path: 继续训练的checkpoint路径
    """
    print("="*60)
    print("PPO微调训练")
    print("="*60)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{output_dir}/eval", exist_ok=True)
    
    print(f"\n配置:")
    print(f"  输出目录: {output_dir}")
    print(f"  总步数: {total_timesteps:,}")
    print(f"  并行环境: {n_envs}")
    print(f"  学习率: {learning_rate}")
    print(f"  n_steps: {n_steps}")
    print(f"  batch_size: {batch_size}")
    print(f"  n_epochs: {n_epochs}")
    
    # 1. 创建训练环境
    print(f"\n{'='*60}")
    print("创建训练环境...")
    print(f"{'='*60}\n")
    
    if n_envs > 1:
        # 多进程并行环境
        env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
        print(f"✅ 创建了 {n_envs} 个并行环境 (SubprocVecEnv)")
    else:
        # 单环境
        env = DummyVecEnv([make_env(0)])
        print(f"✅ 创建了单环境 (DummyVecEnv)")
    
    # 2. 创建评估环境
    eval_env = DummyVecEnv([make_env(100)])
    print(f"✅ 创建了评估环境")
    
    # 3. 加载或创建模型
    print(f"\n{'='*60}")
    print("加载模型...")
    print(f"{'='*60}\n")
    
    if continue_training and checkpoint_path and os.path.exists(checkpoint_path):
        # 继续训练现有模型
        print(f"📂 从checkpoint继续训练: {checkpoint_path}")
        model = PPO.load(checkpoint_path, env=env)
        print(f"✅ Checkpoint加载成功")
        
    elif os.path.exists(bc_model_path):
        # 从BC初始化的PPO开始
        print(f"📂 加载BC初始化的PPO: {bc_model_path}")
        model = PPO.load(bc_model_path, env=env)
        
        # 可选：调整PPO超参数（BC训练时可能用的是默认值）
        model.learning_rate = learning_rate
        model.n_steps = n_steps
        model.batch_size = batch_size
        model.n_epochs = n_epochs
        
        print(f"✅ BC模型加载成功")
        print(f"   已调整训练超参数")
        
    else:
        # 从头开始训练PPO
        print(f"⚠️  未找到BC模型，从头开始训练")
        print(f"   尝试路径: {bc_model_path}")
        
        # 使用与BC一致的网络结构
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64])
        )
        
        model = PPO(
            policy="MlpPolicy",  # Flattened observation使用MLP
            env=env,
            policy_kwargs=policy_kwargs,  # 指定网络结构
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            use_sde=False,
            sde_sample_freq=-1,
            target_kl=None,
            tensorboard_log=f"{output_dir}/tensorboard",
            verbose=1,
        )
        print(f"✅ 新PPO模型创建成功 (net_arch=[256, 128, 64])")
    
    # 4. 设置callbacks
    print(f"\n{'='*60}")
    print("设置训练callbacks...")
    print(f"{'='*60}\n")
    
    callbacks = []
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=f"{output_dir}/checkpoints",
        name_prefix="ppo_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)
    print(f"✅ Checkpoint每 {save_freq:,} 步保存一次")
    
    # Eval callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{output_dir}/eval",
        log_path=f"{output_dir}/eval",
        eval_freq=eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )
    callbacks.append(eval_callback)
    print(f"✅ 评估每 {eval_freq:,} 步执行一次")
    
    # 5. 开始训练
    print(f"\n{'='*60}")
    print(f"开始PPO训练...")
    print(f"{'='*60}\n")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=10,
            tb_log_name="ppo_run",
            reset_num_timesteps=not continue_training,
            progress_bar=True,
        )
        
        print(f"\n{'='*60}")
        print("✅ 训练完成！")
        print(f"{'='*60}\n")
        
    except KeyboardInterrupt:
        print(f"\n{'='*60}")
        print("⚠️  训练被用户中断")
        print(f"{'='*60}\n")
    
    # 6. 保存最终模型
    final_model_path = f"{output_dir}/ppo_finetuned.zip"
    model.save(final_model_path)
    print(f"✅ 最终模型已保存: {final_model_path}")
    
    # 7. 总结
    print(f"\n{'='*60}")
    print("训练完成总结")
    print(f"{'='*60}")
    print(f"✅ 最终模型: {final_model_path}")
    print(f"✅ 最佳模型: {output_dir}/eval/best_model.zip")
    print(f"✅ Checkpoints: {output_dir}/checkpoints/")
    print(f"✅ TensorBoard日志: {output_dir}/tensorboard/")
    print(f"\n查看训练曲线:")
    print(f"  tensorboard --logdir {output_dir}/tensorboard")
    print(f"\n使用模型:")
    print(f"  修改robot.py中的PPO_FINETUNED_PATH = '{final_model_path}'")
    print(f"{'='*60}\n")
    
    # 清理
    env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO微调训练")
    
    # 模型路径
    parser.add_argument(
        "--bc_model", 
        type=str, 
        default="./bc_models_imitation/bc_ppo_ready.zip",
        help="BC初始化的PPO模型路径"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./ppo_models",
        help="输出目录"
    )
    
    # 训练参数
    parser.add_argument(
        "--timesteps", 
        type=int, 
        default=1_000_000,
        help="总训练步数"
    )
    parser.add_argument(
        "--n_envs", 
        type=int, 
        default=4,
        help="并行环境数量"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=3e-4,
        help="学习率"
    )
    parser.add_argument(
        "--n_steps", 
        type=int, 
        default=2048,
        help="每次更新的步数"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=64,
        help="批大小"
    )
    parser.add_argument(
        "--n_epochs", 
        type=int, 
        default=10,
        help="PPO更新epochs"
    )
    
    # Callback参数
    parser.add_argument(
        "--save_freq", 
        type=int, 
        default=50_000,
        help="Checkpoint保存频率"
    )
    parser.add_argument(
        "--eval_freq", 
        type=int, 
        default=25_000,
        help="评估频率"
    )
    
    # 继续训练
    parser.add_argument(
        "--continue_training",
        action="store_true",
        help="继续训练现有模型"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="继续训练的checkpoint路径"
    )
    
    args = parser.parse_args()
    
    # 训练
    train_ppo(
        bc_model_path=args.bc_model,
        output_dir=args.output_dir,
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
        continue_training=args.continue_training,
        checkpoint_path=args.checkpoint,
    )

