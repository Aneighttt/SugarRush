"""
测试BC模型和PPO的兼容性
"""
import torch
import os
from stable_baselines3 import PPO
from environment import BomberEnv
from gymnasium.wrappers import FlattenObservation

def test_compatibility():
    print("="*60)
    print("测试BC模型与PPO的兼容性")
    print("="*60)
    
    # 1. 检查BC模型是否存在
    bc_policy_path = "./bc_models_imitation/bc_policy.pt"
    ppo_path = "./bc_models_imitation/bc_ppo_ready.zip"
    
    print("\n检查文件...")
    if os.path.exists(bc_policy_path):
        print(f"✅ BC Policy存在: {bc_policy_path}")
    else:
        print(f"❌ BC Policy不存在: {bc_policy_path}")
        return
    
    if os.path.exists(ppo_path):
        print(f"✅ PPO模型存在: {ppo_path}")
        has_ppo = True
    else:
        print(f"⚠️  PPO模型不存在: {ppo_path}")
        has_ppo = False
    
    # 2. 测试加载BC Policy
    print("\n测试加载BC Policy...")
    try:
        bc_policy = torch.load(bc_policy_path, map_location='cpu')
        print(f"✅ BC Policy加载成功")
        print(f"   类型: {type(bc_policy)}")
    except Exception as e:
        print(f"❌ BC Policy加载失败: {e}")
        return
    
    # 3. 测试PPO模型
    if has_ppo:
        print("\n测试加载PPO模型...")
        try:
            # 需要使用flattened环境
            base_env = BomberEnv()
            env = FlattenObservation(base_env)
            
            ppo_model = PPO.load(ppo_path, env=env)
            print(f"✅ PPO模型加载成功")
            
            # 测试预测
            obs = env.reset()
            action, _ = ppo_model.predict(obs, deterministic=True)
            print(f"✅ PPO预测测试成功")
            print(f"   观察shape: {obs.shape}")
            print(f"   动作: {action}")
            
        except Exception as e:
            print(f"❌ PPO模型测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 4. 总结
    print("\n" + "="*60)
    print("兼容性测试结果")
    print("="*60)
    
    if has_ppo:
        print("✅ BC模型已保存")
        print("✅ PPO模型已保存（可能需要调整）")
        print("\n⚠️  重要提示:")
        print("   PPO使用时必须用FlattenObservation包装环境:")
        print("   ```python")
        print("   from gymnasium.wrappers import FlattenObservation")
        print("   env = FlattenObservation(BomberEnv())")
        print("   model = PPO.load('bc_ppo_ready.zip', env=env)")
        print("   ```")
    else:
        print("⚠️  PPO模型未保存，需要重新训练")
    
    print("="*60)


if __name__ == "__main__":
    test_compatibility()

