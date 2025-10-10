"""
PPO智能体模块 - 用于从BC预训练模型进行微调
支持加载BC预训练的策略网络
"""
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
from typing import Dict
import numpy as np

from bc_trainer import BCPolicyNetwork


class CustomCNN(BaseFeaturesExtractor):
    """
    自定义特征提取器，用于处理grid_view
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[0]  # 13 channels
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # 计算CNN输出维度
        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape)
            n_flatten = self.cnn(sample).shape[1]
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


class CustomMultiInputPolicy(ActorCriticPolicy):
    """
    自定义多输入策略，用于处理Dict观察空间
    """
    def __init__(self, *args, **kwargs):
        super(CustomMultiInputPolicy, self).__init__(
            *args,
            **kwargs,
        )


class PPOAgent:
    """
    PPO智能体，支持从BC模型初始化
    """
    
    def __init__(self, env, bc_model_path: str = None, 
                 learning_rate: float = 3e-4, 
                 n_steps: int = 2048,
                 batch_size: int = 64,
                 n_epochs: int = 10,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_range: float = 0.2,
                 ent_coef: float = 0.01,
                 vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 tensorboard_log: str = "./ppo_logs",
                 device: str = 'auto'):
        """
        初始化PPO智能体
        
        Args:
            env: Gym环境
            bc_model_path: BC预训练模型路径（可选）
            learning_rate: 学习率
            n_steps: 每次更新收集的步数
            batch_size: 批次大小
            n_epochs: 每次更新的训练轮数
            gamma: 折扣因子
            gae_lambda: GAE参数
            clip_range: PPO裁剪范围
            ent_coef: 熵系数
            vf_coef: 价值函数系数
            max_grad_norm: 梯度裁剪
            tensorboard_log: TensorBoard日志目录
            device: 训练设备
        """
        
        # 配置策略网络参数
        policy_kwargs = dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=[dict(pi=[128, 128], vf=[128, 128])]
        )
        
        # 创建PPO模型
        self.model = PPO(
            policy="MultiInputPolicy",
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            verbose=1,
            tensorboard_log=tensorboard_log,
            device=device,
            policy_kwargs=policy_kwargs
        )
        
        # 如果提供了BC模型，加载预训练权重
        if bc_model_path is not None:
            self.load_bc_weights(bc_model_path)
    
    def load_bc_weights(self, bc_model_path: str):
        """
        从BC模型加载策略网络权重
        
        Args:
            bc_model_path: BC模型路径
        """
        print(f"从BC模型加载权重: {bc_model_path}")
        
        # 加载BC模型
        checkpoint = torch.load(bc_model_path, map_location=self.model.device)
        bc_state_dict = checkpoint['model_state_dict']
        
        # 获取PPO策略网络
        ppo_policy = self.model.policy
        
        # 尝试匹配和加载权重
        # 注意：BC模型和PPO模型的结构可能不完全相同，需要仔细匹配
        
        # 方法1: 直接加载兼容的层
        try:
            # 提取特征提取器（CNN部分）
            for bc_key, bc_param in bc_state_dict.items():
                if bc_key.startswith('conv_layers'):
                    # 映射到PPO的features_extractor
                    ppo_key = bc_key.replace('conv_layers', 'features_extractor.cnn')
                    if ppo_key in ppo_policy.state_dict():
                        ppo_policy.state_dict()[ppo_key].copy_(bc_param)
                        print(f"加载层: {bc_key} -> {ppo_key}")
            
            # 如果BC模型的player_mlp和fusion层可以映射，也加载它们
            # 这部分需要根据具体的网络结构调整
            
            print("BC权重加载完成！")
        except Exception as e:
            print(f"加载BC权重时出现问题: {e}")
            print("将从头开始训练PPO策略网络")
    
    def train(self, total_timesteps: int, log_interval: int = 10):
        """
        训练PPO智能体
        
        Args:
            total_timesteps: 总训练步数
            log_interval: 日志记录间隔
        """
        self.model.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval
        )
    
    def save(self, path: str):
        """保存模型"""
        self.model.save(path)
        print(f"模型已保存到: {path}")
    
    def load(self, path: str):
        """加载模型"""
        self.model = PPO.load(path, env=self.model.get_env(), device=self.model.device)
        print(f"模型已从 {path} 加载")
    
    def predict(self, observation, deterministic: bool = False):
        """预测动作"""
        return self.model.predict(observation, deterministic=deterministic)


class BC2PPOConverter:
    """
    将BC模型转换为PPO模型的工具类
    提供更灵活的权重转换方法
    """
    
    @staticmethod
    def create_ppo_from_bc(bc_model: BCPolicyNetwork, env, device='auto'):
        """
        从BC模型创建PPO智能体，并尽可能复用BC的权重
        
        Args:
            bc_model: 训练好的BC模型
            env: Gym环境
            device: 设备
            
        Returns:
            PPOAgent实例
        """
        # 创建PPO智能体（不加载BC权重）
        ppo_agent = PPOAgent(env, bc_model_path=None, device=device)
        
        # 手动复制BC模型的权重到PPO的策略网络
        ppo_policy = ppo_agent.model.policy
        bc_state = bc_model.state_dict()
        
        # 映射权重
        # 这里需要根据实际的网络结构进行调整
        mapping = {
            # BC的卷积层 -> PPO的特征提取器
            'conv_layers.0.weight': 'features_extractor.cnn.0.weight',
            'conv_layers.0.bias': 'features_extractor.cnn.0.bias',
            'conv_layers.2.weight': 'features_extractor.cnn.2.weight',
            'conv_layers.2.bias': 'features_extractor.cnn.2.bias',
            'conv_layers.4.weight': 'features_extractor.cnn.4.weight',
            'conv_layers.4.bias': 'features_extractor.cnn.4.bias',
        }
        
        ppo_state = ppo_policy.state_dict()
        
        for bc_key, ppo_key in mapping.items():
            if bc_key in bc_state and ppo_key in ppo_state:
                if bc_state[bc_key].shape == ppo_state[ppo_key].shape:
                    ppo_state[ppo_key].copy_(bc_state[bc_key])
                    print(f"复制权重: {bc_key} -> {ppo_key}")
                else:
                    print(f"形状不匹配: {bc_key} {bc_state[bc_key].shape} vs {ppo_key} {ppo_state[ppo_key].shape}")
        
        ppo_policy.load_state_dict(ppo_state)
        
        print("BC模型权重已转换到PPO模型！")
        return ppo_agent


if __name__ == "__main__":
    # 使用示例
    from environment import BomberEnv
    
    # 创建环境
    env = BomberEnv()
    
    # 方式1: 直接创建PPO智能体（从BC模型初始化）
    ppo_agent = PPOAgent(
        env=env,
        bc_model_path="./bc_models/bc_best_model.pt",
        learning_rate=3e-4,
        device='cpu'
    )
    
    # 训练
    # ppo_agent.train(total_timesteps=100000)
    
    # 保存
    # ppo_agent.save("./ppo_models/ppo_finetuned.zip")

