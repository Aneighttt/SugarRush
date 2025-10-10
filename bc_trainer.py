"""
行为克隆（Behavior Cloning）训练模块
用于从专家数据训练策略网络
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, Tuple
import os
from datetime import datetime
from tqdm import tqdm

from bc_data_collector import ExpertDataLoader


class ExpertDataset(Dataset):
    """专家数据的PyTorch数据集"""
    
    def __init__(self, observations: Dict[str, np.ndarray], actions: np.ndarray):
        """
        Args:
            observations: 观察字典 {"grid_view": array, "player_state": array}
            actions: 动作数组
        """
        self.grid_views = torch.FloatTensor(observations["grid_view"])
        self.player_states = torch.FloatTensor(observations["player_state"])
        self.actions = torch.LongTensor(actions)
        
        assert len(self.grid_views) == len(self.player_states) == len(self.actions)
        
    def __len__(self):
        return len(self.actions)
    
    def __getitem__(self, idx):
        return {
            "grid_view": self.grid_views[idx],
            "player_state": self.player_states[idx],
            "action": self.actions[idx]
        }


class BCPolicyNetwork(nn.Module):
    """
    行为克隆策略网络（支持MultiDiscrete动作空间）
    输入: 观察（grid_view + player_state）
    输出: 三个独立的动作概率分布（方向、炸弹、速度）
    """
    
    def __init__(self, grid_channels=14, grid_height=16, grid_width=28, 
                 player_state_dim=10):
        super(BCPolicyNetwork, self).__init__()
        
        # CNN for grid_view (14 x 16 x 28)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(grid_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> 64 x 8 x 14
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # -> 64 x 4 x 7
            nn.ReLU(),
        )
        
        # 计算展平后的维度
        conv_out_size = 64 * 4 * 7  # 1792
        
        # MLP for player_state
        self.player_mlp = nn.Sequential(
            nn.Linear(player_state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # 共享融合层
        self.shared_fusion = nn.Sequential(
            nn.Linear(conv_out_size + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # MultiDiscrete输出头: [方向(5), 炸弹(2), 速度(5)]
        self.direction_head = nn.Linear(128, 5)  # 5个方向
        self.bomb_head = nn.Linear(128, 2)       # 是否放炸弹
        self.speed_head = nn.Linear(128, 5)      # 5个速度档位
        
    def forward(self, grid_view, player_state):
        # 处理grid_view
        x_grid = self.conv_layers(grid_view)
        x_grid = x_grid.view(x_grid.size(0), -1)  # 展平
        
        # 处理player_state
        x_player = self.player_mlp(player_state)
        
        # 融合
        x = torch.cat([x_grid, x_player], dim=1)
        x = self.shared_fusion(x)
        
        # 三个独立的输出
        direction_logits = self.direction_head(x)
        bomb_logits = self.bomb_head(x)
        speed_logits = self.speed_head(x)
        
        return direction_logits, bomb_logits, speed_logits


class BCTrainer:
    """行为克隆训练器"""
    
    def __init__(self, model: BCPolicyNetwork, device='cpu', learning_rate=1e-3):
        """
        Args:
            model: 策略网络
            device: 训练设备
            learning_rate: 学习率
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct_direction = 0
        correct_bomb = 0
        correct_speed = 0
        total = 0
        
        for batch in tqdm(dataloader, desc="Training"):
            grid_view = batch["grid_view"].to(self.device)
            player_state = batch["player_state"].to(self.device)
            actions = batch["action"].to(self.device)  # Shape: (batch_size, 3)
            
            # 分离三个动作维度
            direction_actions = actions[:, 0]  # 方向
            bomb_actions = actions[:, 1]       # 炸弹
            speed_actions = actions[:, 2]      # 速度
            
            # 前向传播
            direction_logits, bomb_logits, speed_logits = self.model(grid_view, player_state)
            
            # 计算三个独立的损失
            loss_direction = self.criterion(direction_logits, direction_actions)
            loss_bomb = self.criterion(bomb_logits, bomb_actions)
            loss_speed = self.criterion(speed_logits, speed_actions)
            
            # 总损失（可以加权重，这里使用相等权重）
            loss = loss_direction + loss_bomb + loss_speed
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            
            # 计算每个动作维度的准确率
            _, pred_direction = torch.max(direction_logits, 1)
            _, pred_bomb = torch.max(bomb_logits, 1)
            _, pred_speed = torch.max(speed_logits, 1)
            
            correct_direction += (pred_direction == direction_actions).sum().item()
            correct_bomb += (pred_bomb == bomb_actions).sum().item()
            correct_speed += (pred_speed == speed_actions).sum().item()
            total += actions.size(0)
        
        avg_loss = total_loss / len(dataloader)
        # 计算平均准确率（三个维度的平均）
        accuracy = (correct_direction + correct_bomb + correct_speed) / (3 * total)
        
        return avg_loss, accuracy
    
    def validate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """验证"""
        self.model.eval()
        total_loss = 0
        correct_direction = 0
        correct_bomb = 0
        correct_speed = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                grid_view = batch["grid_view"].to(self.device)
                player_state = batch["player_state"].to(self.device)
                actions = batch["action"].to(self.device)  # Shape: (batch_size, 3)
                
                # 分离三个动作维度
                direction_actions = actions[:, 0]
                bomb_actions = actions[:, 1]
                speed_actions = actions[:, 2]
                
                # 前向传播
                direction_logits, bomb_logits, speed_logits = self.model(grid_view, player_state)
                
                # 计算三个独立的损失
                loss_direction = self.criterion(direction_logits, direction_actions)
                loss_bomb = self.criterion(bomb_logits, bomb_actions)
                loss_speed = self.criterion(speed_logits, speed_actions)
                
                loss = loss_direction + loss_bomb + loss_speed
                
                # 统计
                total_loss += loss.item()
                
                _, pred_direction = torch.max(direction_logits, 1)
                _, pred_bomb = torch.max(bomb_logits, 1)
                _, pred_speed = torch.max(speed_logits, 1)
                
                correct_direction += (pred_direction == direction_actions).sum().item()
                correct_bomb += (pred_bomb == bomb_actions).sum().item()
                correct_speed += (pred_speed == speed_actions).sum().item()
                total += actions.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = (correct_direction + correct_bomb + correct_speed) / (3 * total)
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader = None, 
              num_epochs: int = 50, save_dir: str = "./bc_models"):
        """
        完整训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
            save_dir: 模型保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*60}")
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            print(f"训练 - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
            
            # 验证
            if val_loader is not None:
                val_loss, val_acc = self.validate(val_loader)
                self.val_losses.append(val_loss)
                self.val_accs.append(val_acc)
                
                print(f"验证 - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
                
                # 学习率调度
                self.scheduler.step(val_loss)
                
                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_path = os.path.join(save_dir, "bc_best_model.pt")
                    self.save_model(best_model_path)
                    print(f"保存最佳模型到: {best_model_path}")
            
            # 定期保存检查点
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(save_dir, f"bc_checkpoint_epoch{epoch+1}.pt")
                self.save_model(checkpoint_path)
        
        # 保存最终模型
        final_model_path = os.path.join(save_dir, "bc_final_model.pt")
        self.save_model(final_model_path)
        print(f"\n训练完成！最终模型保存到: {final_model_path}")
        
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs
        }, path)
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_accs = checkpoint.get('train_accs', [])
        self.val_accs = checkpoint.get('val_accs', [])
        print(f"模型已从 {path} 加载")


def train_bc_from_data(data_path: str, val_split: float = 0.1, batch_size: int = 128,
                       num_epochs: int = 50, learning_rate: float = 1e-3, device: str = 'cpu'):
    """
    从专家数据训练BC模型的便捷函数
    
    Args:
        data_path: 专家数据路径（文件或目录）
        val_split: 验证集比例
        batch_size: 批次大小
        num_epochs: 训练轮数
        learning_rate: 学习率
        device: 训练设备
    """
    print("加载专家数据...")
    loader = ExpertDataLoader(data_path)
    loader.load_data()
    loader.print_statistics()
    
    # 获取数据
    observations, actions = loader.get_transitions_as_arrays()
    
    # 划分训练集和验证集
    total_samples = len(actions)
    val_size = int(total_samples * val_split)
    train_size = total_samples - val_size
    
    indices = np.random.permutation(total_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_obs = {
        "grid_view": observations["grid_view"][train_indices],
        "player_state": observations["player_state"][train_indices]
    }
    train_actions = actions[train_indices]
    
    val_obs = {
        "grid_view": observations["grid_view"][val_indices],
        "player_state": observations["player_state"][val_indices]
    }
    val_actions = actions[val_indices]
    
    # 创建数据集和数据加载器
    train_dataset = ExpertDataset(train_obs, train_actions)
    val_dataset = ExpertDataset(val_obs, val_actions)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"\n训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 创建模型和训练器
    model = BCPolicyNetwork()
    trainer = BCTrainer(model, device=device, learning_rate=learning_rate)
    
    # 训练
    trainer.train(train_loader, val_loader, num_epochs=num_epochs)
    
    return model, trainer


if __name__ == "__main__":
    # 使用示例
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    model, trainer = train_bc_from_data(
        data_path="./expert_data",
        val_split=0.1,
        batch_size=128,
        num_epochs=50,
        learning_rate=1e-3,
        device=device
    )

