"""
单独的BC到PPO转换脚本
用于已训练完成的BC模型
"""
import torch
import os
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from environment import BomberEnv
from gymnasium.wrappers import FlattenObservation


def convert_bc_to_ppo(
    bc_policy_path: str,
    output_path: str,
    net_arch: list = [256, 128, 64],
    learning_rate: float = 3e-4,
):
    """
    将BC policy转换为PPO格式
    
    Args:
        bc_policy_path: BC policy路径 (.pt 或 .pth)
        output_path: PPO输出路径 (.zip)
        net_arch: 网络结构
        learning_rate: 学习率
    """
    print("="*60)
    print("BC模型转PPO格式")
    print("="*60)
    print(f"\n输入: {bc_policy_path}")
    print(f"输出: {output_path}")
    print(f"网络结构: {net_arch}\n")
    
    # 检查输入文件
    if not os.path.exists(bc_policy_path):
        raise FileNotFoundError(f"BC模型不存在: {bc_policy_path}")
    
    # 创建环境
    print("创建环境...")
    base_env = BomberEnv()
    env = FlattenObservation(base_env)
    print(f"✅ 环境已创建")
    print(f"   Observation: {env.observation_space}")
    print(f"   Action: {env.action_space}\n")
    
    # 加载BC policy
    print("加载BC policy...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if bc_policy_path.endswith('.pt'):
        # 完整对象
        bc_policy = torch.load(bc_policy_path, map_location=device, weights_only=False)
        bc_state_dict = bc_policy.state_dict()
    else:
        # 只有state_dict
        bc_state_dict = torch.load(bc_policy_path, map_location=device, weights_only=False)
    
    print(f"✅ BC policy已加载 ({len(bc_state_dict)} 参数)\n")
    
    # 创建PPO模型
    print("创建PPO模型...")
    policy_kwargs = dict(
        net_arch=dict(pi=net_arch, vf=net_arch)
    )
    
    ppo_model = PPO(
        policy=ActorCriticPolicy,
        env=env,
        policy_kwargs=policy_kwargs,
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
    print(f"✅ PPO模型已创建 (net_arch={net_arch})\n")
    
    # 加载BC参数到PPO
    print("转移BC参数到PPO...")
    try:
        # 检查参数匹配
        bc_params = set(bc_state_dict.keys())
        ppo_params = set(ppo_model.policy.state_dict().keys())
        
        matched = bc_params & ppo_params
        bc_only = bc_params - ppo_params
        ppo_only = ppo_params - bc_params
        
        print(f"  匹配的参数: {len(matched)}/{len(bc_params)}")
        
        if bc_only:
            print(f"  BC独有参数 ({len(bc_only)}): {list(bc_only)[:3]}...")
        if ppo_only:
            print(f"  PPO独有参数 ({len(ppo_only)}): {list(ppo_only)[:3]}...")
        
        # 加载
        ppo_model.policy.load_state_dict(bc_state_dict, strict=False)
        print(f"✅ 参数转移成功\n")
        
    except Exception as e:
        print(f"❌ 参数转移失败: {e}")
        raise
    
    # 保存PPO模型
    print(f"保存PPO模型到: {output_path}")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    ppo_model.save(output_path)
    print(f"✅ 保存成功\n")
    
    # 总结
    print("="*60)
    print("转换完成总结")
    print("="*60)
    print(f"✅ PPO模型: {output_path}")
    print(f"   网络结构: {net_arch}")
    print(f"   参数数量: {len(matched)}")
    print(f"\n使用方法:")
    print(f"   在 robot.py 中会自动加载此模型")
    print(f"   或用于PPO微调:")
    print(f"     python train_ppo.py --bc_model {output_path}")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="BC模型转PPO格式")
    parser.add_argument(
        "--bc_policy",
        type=str,
        required=True,
        help="BC policy路径 (.pt 或 .pth)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="PPO输出路径 (.zip)"
    )
    parser.add_argument(
        "--net_arch",
        type=int,
        nargs="+",
        default=[256, 128, 64],
        help="网络结构 (默认: 256 128 64)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="学习率 (默认: 3e-4)"
    )
    
    args = parser.parse_args()
    
    convert_bc_to_ppo(
        bc_policy_path=args.bc_policy,
        output_path=args.output,
        net_arch=args.net_arch,
        learning_rate=args.lr,
    )

