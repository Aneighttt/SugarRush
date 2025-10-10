"""
数据收集模块 - 用于记录专家AI的观察和动作对
支持离线数据收集和实时数据收集两种模式
"""
import numpy as np
import pickle
import os
from datetime import datetime
from typing import Dict, List, Tuple
import json


class ExpertDataCollector:
    """收集专家AI的状态-动作对数据"""
    
    def __init__(self, save_dir: str = "./expert_data"):
        """
        初始化数据收集器
        
        Args:
            save_dir: 数据保存目录
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 存储当前episode的数据
        self.current_episode = {
            "observations": [],  # 观察
            "actions": [],       # 动作 (MultiDiscrete)
            "next_observations": [],  # 下一个观察
            "rewards": [],       # 奖励（可选，BC不需要）
            "dones": []          # 是否结束
        }
        
        # 统计信息
        self.stats = {
            "total_episodes": 0,
            "total_transitions": 0,
            "action_distribution": {
                "direction": {i: 0 for i in range(5)},  # 方向分布
                "bomb": {0: 0, 1: 0},                    # 放炸弹分布
                "speed": {i: 0 for i in range(5)}       # 速度档位分布
            }
        }
        
        self.episode_buffer = []  # 存储多个episode
        
    def add_transition(self, obs: Dict, action: np.ndarray, next_obs: Dict, 
                      reward: float = 0.0, done: bool = False):
        """
        添加一个transition（状态转换）
        
        Args:
            obs: 当前观察（字典格式: {"grid_view": ..., "player_state": ...}）
            action: 执行的动作（MultiDiscrete: [direction, bomb, speed]）
            next_obs: 下一个观察
            reward: 奖励值（可选）
            done: 是否结束
        """
        self.current_episode["observations"].append(obs)
        self.current_episode["actions"].append(action)
        self.current_episode["next_observations"].append(next_obs)
        self.current_episode["rewards"].append(reward)
        self.current_episode["dones"].append(done)
        
        # 更新统计（MultiDiscrete动作）
        if isinstance(action, np.ndarray) and len(action) == 3:
            direction, bomb, speed = action
            self.stats["action_distribution"]["direction"][int(direction)] += 1
            self.stats["action_distribution"]["bomb"][int(bomb)] += 1
            self.stats["action_distribution"]["speed"][int(speed)] += 1
        else:
            # 兼容旧的单一动作格式
            if isinstance(action, (int, np.integer)):
                # 这是旧格式，不更新统计
                pass
        
        self.stats["total_transitions"] += 1
        
    def finish_episode(self):
        """完成当前episode，将数据加入buffer"""
        if len(self.current_episode["observations"]) > 0:
            self.episode_buffer.append(self.current_episode.copy())
            self.stats["total_episodes"] += 1
            
            # 重置当前episode
            self.current_episode = {
                "observations": [],
                "actions": [],
                "next_observations": [],
                "rewards": [],
                "dones": []
            }
            
    def save_data(self, filename: str = None):
        """
        保存收集的数据到文件
        
        Args:
            filename: 文件名（如果为None，自动生成带时间戳的文件名）
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"expert_data_{timestamp}.pkl"
        
        filepath = os.path.join(self.save_dir, filename)
        
        # 准备保存的数据
        save_data = {
            "episodes": self.episode_buffer,
            "stats": self.stats
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        # 保存统计信息（JSON格式便于查看）
        stats_file = filepath.replace('.pkl', '_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        print(f"✅ 数据已保存到: {filepath}")
        print(f"✅ 统计信息已保存到: {stats_file}")
        print(f"   总episode数: {self.stats['total_episodes']}")
        print(f"   总transition数: {self.stats['total_transitions']}")
        
        # 打印MultiDiscrete动作分布
        if "direction" in self.stats["action_distribution"]:
            print(f"\n   动作分布:")
            direction_names = ["不动", "上", "下", "左", "右"]
            for i, count in self.stats["action_distribution"]["direction"].items():
                if count > 0:
                    print(f"     方向-{direction_names[i]}: {count} ({count/self.stats['total_transitions']*100:.1f}%)")
            
            for i, count in self.stats["action_distribution"]["bomb"].items():
                if count > 0:
                    bomb_name = "不放炸弹" if i == 0 else "放炸弹"
                    print(f"     {bomb_name}: {count} ({count/self.stats['total_transitions']*100:.1f}%)")
            
            speed_names = ["最大速度", "极慢", "慢速", "中速", "快速"]
            for i, count in self.stats["action_distribution"]["speed"].items():
                if count > 0:
                    print(f"     速度-{speed_names[i]}: {count} ({count/self.stats['total_transitions']*100:.1f}%)")
        
        return filepath
    
    def clear_buffer(self):
        """清空buffer"""
        self.episode_buffer = []
        self.current_episode = {
            "observations": [],
            "actions": [],
            "next_observations": [],
            "rewards": [],
            "dones": []
        }


class ExpertDataLoader:
    """加载和处理专家数据的工具类"""
    
    def __init__(self, data_path: str):
        """
        初始化数据加载器
        
        Args:
            data_path: 数据文件路径或目录路径
        """
        self.data_path = data_path
        self.episodes = []
        self.stats = {}
        
    def load_data(self):
        """加载数据文件"""
        if os.path.isdir(self.data_path):
            # 如果是目录，加载所有.pkl文件
            pkl_files = [f for f in os.listdir(self.data_path) if f.endswith('.pkl')]
            for pkl_file in pkl_files:
                self._load_single_file(os.path.join(self.data_path, pkl_file))
        else:
            # 单个文件
            self._load_single_file(self.data_path)
        
        print(f"总共加载了 {len(self.episodes)} 个episodes")
        
    def _load_single_file(self, filepath: str):
        """加载单个数据文件"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.episodes.extend(data["episodes"])
        
        # 合并统计信息
        if not self.stats:
            self.stats = data["stats"]
        else:
            self.stats["total_episodes"] += data["stats"]["total_episodes"]
            self.stats["total_transitions"] += data["stats"]["total_transitions"]
            for action, count in data["stats"]["action_distribution"].items():
                self.stats["action_distribution"][action] += count
    
    def get_all_transitions(self) -> Tuple[List, List]:
        """
        获取所有transitions，返回观察和动作列表
        
        Returns:
            observations: 观察列表
            actions: 动作列表
        """
        all_obs = []
        all_actions = []
        
        for episode in self.episodes:
            all_obs.extend(episode["observations"])
            all_actions.extend(episode["actions"])
        
        return all_obs, all_actions
    
    def get_transitions_as_arrays(self) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        将所有transitions转换为numpy数组格式
        
        Returns:
            observations: 字典格式的观察数组 {"grid_view": array, "player_state": array}
            actions: 动作数组
        """
        all_obs, all_actions = self.get_all_transitions()
        
        # 将观察转换为numpy数组
        grid_views = np.array([obs["grid_view"] for obs in all_obs])
        player_states = np.array([obs["player_state"] for obs in all_obs])
        actions = np.array(all_actions)
        
        observations = {
            "grid_view": grid_views,
            "player_state": player_states
        }
        
        return observations, actions
    
    def print_statistics(self):
        """打印数据统计信息"""
        print("=" * 50)
        print("专家数据统计信息")
        print("=" * 50)
        print(f"Episode数量: {self.stats.get('total_episodes', len(self.episodes))}")
        print(f"Transition数量: {self.stats.get('total_transitions', 'N/A')}")
        print(f"\n动作分布:")
        action_names = ["上", "下", "左", "右", "放炸弹", "停止"]
        if "action_distribution" in self.stats:
            total = sum(self.stats["action_distribution"].values())
            for action_id, count in self.stats["action_distribution"].items():
                action_id = int(action_id)
                percentage = (count / total * 100) if total > 0 else 0
                print(f"  动作{action_id} ({action_names[action_id]}): {count} ({percentage:.2f}%)")
        print("=" * 50)


# 使用示例
if __name__ == "__main__":
    # 示例：收集数据
    collector = ExpertDataCollector(save_dir="./expert_data")
    
        # 模拟收集一个episode的数据
    for step in range(100):
        obs = {
            "grid_view": np.random.rand(14, 16, 28).astype(np.float32),
            "player_state": np.random.rand(10).astype(np.float32)
        }
        action = np.random.randint(0, 6)
        next_obs = {
            "grid_view": np.random.rand(14, 16, 28).astype(np.float32),
            "player_state": np.random.rand(10).astype(np.float32)
        }
        
        collector.add_transition(obs, action, next_obs, reward=0.0, done=(step == 99))
    
    collector.finish_episode()
    collector.save_data()
    
    # 示例：加载数据
    loader = ExpertDataLoader("./expert_data")
    loader.load_data()
    loader.print_statistics()

