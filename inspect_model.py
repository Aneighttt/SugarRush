"""
检查模型参数和训练状态
"""
import torch
from stable_baselines3 import PPO
import os


def inspect_model(model_path: str):
    """检查模型详细信息"""
    print("="*60)
    print("模型检查")
    print("="*60)
    print(f"模型路径: {model_path}\n")
    
    if not os.path.exists(model_path):
        print(f"❌ 模型不存在: {model_path}")
        return
    
    # 加载模型
    print("加载模型...")
    try:
        model = PPO.load(model_path, device='cpu')
        print(f"✅ 模型加载成功\n")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return
    
    # 检查网络结构
    print("="*60)
    print("网络结构")
    print("="*60)
    policy = model.policy
    state_dict = policy.state_dict()
    
    print(f"\n总参数数量: {len(state_dict)}")
    
    # 检查关键层的形状
    print("\n关键层形状:")
    key_layers = [
        'mlp_extractor.policy_net.0.weight',
        'mlp_extractor.policy_net.0.bias',
        'mlp_extractor.policy_net.2.weight',
        'mlp_extractor.policy_net.2.bias',
    ]
    
    for key in key_layers:
        if key in state_dict:
            shape = state_dict[key].shape
            print(f"  {key}: {shape}")
    
    # 推断网络结构
    if 'mlp_extractor.policy_net.0.weight' in state_dict:
        first_layer = state_dict['mlp_extractor.policy_net.0.weight']
        input_dim = first_layer.shape[1]
        hidden1 = first_layer.shape[0]
        
        if 'mlp_extractor.policy_net.2.weight' in state_dict:
            hidden2 = state_dict['mlp_extractor.policy_net.2.weight'].shape[0]
            if 'mlp_extractor.policy_net.4.weight' in state_dict:
                hidden3 = state_dict['mlp_extractor.policy_net.4.weight'].shape[0]
                net_arch = f"[{hidden1}, {hidden2}, {hidden3}]"
            else:
                net_arch = f"[{hidden1}, {hidden2}]"
        else:
            net_arch = f"[{hidden1}]"
        
        print(f"\n推断的网络结构:")
        print(f"  输入维度: {input_dim}")
        print(f"  隐藏层: {net_arch}")
    
    # 检查参数是否有更新（非零）
    print("\n="*60)
    print("参数统计")
    print("="*60)
    
    total_params = 0
    zero_params = 0
    
    for key, param in state_dict.items():
        total_params += param.numel()
        zero_params += (param == 0).sum().item()
    
    print(f"\n总参数: {total_params:,}")
    print(f"零参数: {zero_params:,} ({zero_params/total_params*100:.2f}%)")
    
    if zero_params / total_params > 0.5:
        print("⚠️  超过50%的参数为零，模型可能未正确训练")
    
    # 检查参数范围
    print("\n参数值范围:")
    for key in ['mlp_extractor.policy_net.0.weight', 'action_net.weight']:
        if key in state_dict:
            param = state_dict[key]
            print(f"  {key}:")
            print(f"    min={param.min().item():.4f}, max={param.max().item():.4f}")
            print(f"    mean={param.mean().item():.4f}, std={param.std().item():.4f}")
    
    print(f"\n{'='*60}\n")


def check_training_logs():
    """检查训练日志"""
    print("="*60)
    print("检查训练日志")
    print("="*60)
    
    # 查找最近的训练输出
    import subprocess
    
    # 检查是否有训练日志
    log_files = [
        "nohup.out",
        "training.log",
        "bc_training.log",
    ]
    
    found = False
    for log_file in log_files:
        if os.path.exists(log_file):
            print(f"\n找到日志文件: {log_file}")
            print("最后20行包含prob_true_act的记录:")
            print("-"*60)
            
            try:
                result = subprocess.run(
                    f"tail -200 {log_file} | grep -i 'prob_true_act\\|epoch' | tail -20",
                    shell=True,
                    capture_output=True,
                    text=True
                )
                if result.stdout:
                    print(result.stdout)
                    found = True
                else:
                    print("  (未找到相关记录)")
            except Exception as e:
                print(f"  读取失败: {e}")
    
    if not found:
        print("\n⚠️  未找到训练日志")
        print("   无法确认训练效果")
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="检查模型")
    parser.add_argument(
        "--model",
        type=str,
        default="./bc_models_weighted/bc_ppo_weighted.zip",
        help="模型路径"
    )
    parser.add_argument(
        "--logs",
        action="store_true",
        help="检查训练日志"
    )
    
    args = parser.parse_args()
    
    inspect_model(args.model)
    
    if args.logs:
        check_training_logs()

