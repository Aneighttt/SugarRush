"""
实时专家数据收集器
从handle_command接收的frame中提取敌方专家AI的数据
通过前后帧对比推断敌方AI的动作
"""
import numpy as np
from typing import Dict, Optional, Tuple
from data_models import Frame, Player
from frame_processor_multi import preprocess_observation_for_player
from bc_data_collector import ExpertDataCollector
import os


def infer_action_from_frames(prev_player: Player, curr_player: Player, 
                             prev_bombs: list, curr_bombs: list, 
                             prev_frame=None, curr_frame=None) -> np.ndarray:
    """
    从前后帧推断玩家的动作（新的MultiDiscrete格式）
    
    Args:
        prev_player: 上一帧的玩家状态
        curr_player: 当前帧的玩家状态
        prev_bombs: 上一帧的炸弹列表
        curr_bombs: 当前帧的炸弹列表
        prev_frame: 上一帧完整数据（用于计算玩家最大速度）
        curr_frame: 当前帧完整数据（用于检查加速点）
    
    Returns:
        action: numpy数组 [direction, bomb, speed]
            direction: 0=不动, 1=上, 2=下, 3=左, 4=右
            bomb: 0=不放, 1=放
            speed: 0=最大(100%), 1=极慢(20%), 2=慢(40%), 3=中(60%), 4=快(80%)
    """
    from config import BASE_SPEED, SPEED_PER_BOOT
    
    prev_x, prev_y = prev_player.position.x, prev_player.position.y
    curr_x, curr_y = curr_player.position.x, curr_player.position.y
    
    # 检查是否放了炸弹
    prev_bomb_count = sum(1 for b in prev_bombs if b.owner_id == curr_player.id)
    curr_bomb_count = sum(1 for b in curr_bombs if b.owner_id == curr_player.id)
    bomb = 1 if curr_bomb_count > prev_bomb_count else 0
    
    # 计算移动距离和方向
    dx = curr_x - prev_x
    dy = curr_y - prev_y
    distance = np.sqrt(dx**2 + dy**2)
    
    # 计算玩家的最大速度
    max_speed = BASE_SPEED + curr_player.agility_boots_count * SPEED_PER_BOOT
    
    # 检查是否踩在加速点上（如果提供了frame）
    if curr_frame is not None:
        player_pos = curr_player.position
        corners = [
            (player_pos.x - 25, player_pos.y - 25), (player_pos.x + 24, player_pos.y - 25),
            (player_pos.x - 25, player_pos.y + 24), (player_pos.x + 24, player_pos.y + 24)
        ]
        
        on_acceleration = False
        for corner_x, corner_y in corners:
            grid_x = int(corner_x / 50)
            grid_y = int(corner_y / 50)
            if 0 <= grid_y < 16 and 0 <= grid_x < 28:
                terrain = curr_frame.map[grid_y][grid_x].terrain
                if terrain == 'B':
                    on_acceleration = True
                    break
        
        if on_acceleration:
            max_speed *= 2.0
    
    # 推断速度档位（基于移动距离与最大速度的比例）
    # 速度档位：0=100%, 1=20%, 2=40%, 3=60%, 4=80%
    if max_speed > 0:
        speed_ratio = distance / max_speed
        
        # 根据速度比例推断档位
        if speed_ratio > 0.9:  # 接近最大速度
            speed = 0  # 最大速度档位
        elif speed_ratio < 0.3:  # 很慢
            speed = 1  # 极慢档位(20%)
        elif speed_ratio < 0.5:
            speed = 2  # 慢档位(40%)
        elif speed_ratio < 0.7:
            speed = 3  # 中档位(60%)
        else:
            speed = 4  # 快档位(80%)
    else:
        # 如果无法计算最大速度，使用绝对距离
        if distance < 3:
            speed = 1
        elif distance < 6:
            speed = 2
        elif distance < 10:
            speed = 3
        elif distance < 14:
            speed = 4
        else:
            speed = 0
    
    # 推断方向
    threshold = 5  # 像素阈值
    
    if abs(dy) > abs(dx):  # 主要是垂直移动
        if dy < -threshold:
            direction = 1  # 上
        elif dy > threshold:
            direction = 2  # 下
        else:
            direction = 0  # 不动
    else:  # 主要是水平移动
        if dx < -threshold:
            direction = 3  # 左
        elif dx > threshold:
            direction = 4  # 右
        else:
            direction = 0  # 不动
    
    return np.array([direction, bomb, speed], dtype=np.int64)


class RealtimeExpertCollector:
    """
    实时收集敌方专家AI的数据
    只需要从友方AI收到的frame中提取信息
    支持自动检测游戏结束并保存
    """
    
    def __init__(self, save_dir: str = "./expert_data", save_interval: int = 100, 
                 auto_save: bool = True, max_ticks: int = 1800):
        """
        Args:
            save_dir: 数据保存目录
            save_interval: 每隔多少帧打印统计（用于监控）
            auto_save: 是否自动保存（检测到游戏结束时）
            max_ticks: 游戏最大tick数（默认1800），接近此值时自动保存
        """
        self.save_dir = save_dir
        self.save_interval = save_interval
        self.auto_save = auto_save
        self.max_ticks = max_ticks
        os.makedirs(save_dir, exist_ok=True)
        
        # 为每个专家AI创建收集器
        self.collectors: Dict[str, ExpertDataCollector] = {}
        
        # 存储上一帧的信息
        self.last_frame: Optional[Frame] = None
        self.last_observations: Dict[str, dict] = {}
        
        # 统计
        self.frame_count = 0
        self.episode_count = 0
        self.current_match_id = None
        self.last_tick = 0
        
        print(f"✅ 实时专家数据收集器已初始化")
        print(f"   自动保存: {'启用' if auto_save else '禁用'}")
        print(f"   游戏最大tick: {max_ticks}")
    
    def process_frame(self, frame: Frame) -> Dict[str, np.ndarray]:
        """
        处理接收到的frame，提取敌方专家数据
        
        Args:
            frame: 从handle_command接收到的frame
            
        Returns:
            推断的专家动作字典 {player_id: action_array}
            action_array是numpy数组 [direction, bomb, speed]
        """
        # 检查是否是新的match（新局游戏）
        if self.current_match_id != frame.current_match_id:
            if self.current_match_id is not None:
                # 上一局结束，保存数据
                self.finish_episode()
            
            self.current_match_id = frame.current_match_id
            self.episode_count += 1
            self.frame_count = 0
            print(f"\n开始新局游戏: {self.current_match_id} (Episode {self.episode_count})")
        
        expert_actions = {}
        
        # 只在有上一帧时才能推断动作
        if self.last_frame is not None:
            # 遍历当前帧的other_players，找出敌方专家
            for curr_player in frame.other_players:
                # 只处理敌方AI（Team B）
                if curr_player.team == frame.my_player.team:
                    continue  # 跳过友方
                
                # 初始化该专家的收集器
                if curr_player.id not in self.collectors:
                    expert_save_dir = f"{self.save_dir}/expert_{curr_player.id}"
                    self.collectors[curr_player.id] = ExpertDataCollector(
                        save_dir=expert_save_dir
                    )
                    print(f"初始化敌方专家AI: {curr_player.id} (Team: {curr_player.team})")
                
                # 在上一帧中找到同一个玩家
                prev_player = None
                for p in self.last_frame.other_players:
                    if p.id == curr_player.id:
                        prev_player = p
                        break
                
                if prev_player is None:
                    continue  # 该玩家在上一帧不存在（可能刚加入）
                
                # 推断该专家的动作（传入完整frame用于计算相对速度）
                action = infer_action_from_frames(
                    prev_player, curr_player,
                    self.last_frame.bombs, frame.bombs,
                    self.last_frame, frame
                )
                expert_actions[curr_player.id] = action
                
                # 创建该专家视角的观察
                curr_obs = preprocess_observation_for_player(frame, curr_player)
                
                # 如果有上一次的观察，保存transition
                if curr_player.id in self.last_observations:
                    prev_obs = self.last_observations[curr_player.id]
                    
                    self.collectors[curr_player.id].add_transition(
                        obs=prev_obs,
                        action=action,
                        next_obs=curr_obs,
                        reward=0.0,
                        done=False
                    )
                
                # 更新观察
                self.last_observations[curr_player.id] = curr_obs
        
        # 更新last_frame
        self.last_frame = frame
        self.frame_count += 1
        self.last_tick = frame.current_tick  # 更新当前tick
        
        # 定期打印进度（不保存）
        if self.frame_count % self.save_interval == 0:
            self.print_progress()
        
        return expert_actions
    
    def finish_episode(self):
        """完成当前episode并保存"""
        if self.frame_count == 0:
            print("⚠️  没有数据可保存（frame_count=0）")
            return
            
        print(f"\n{'='*60}")
        print(f"💾 保存Episode {self.episode_count}")
        print(f"   总帧数: {self.frame_count}")
        print(f"   最后Tick: {self.last_tick}")
        
        for player_id, collector in self.collectors.items():
            collector.finish_episode()
        
        # 保存数据
        self.save_all(self.episode_count)
        
        # 打印总进度（在递增之前）
        print(f"\n📊 总进度：已保存 {self.episode_count} 个Episode")
        print(f"{'='*60}\n")
        
        # 递增episode计数（为下一局准备，但当前游戏可能已经开始了）
        # 注意：下一局的episode_count会在process_frame中重新设置
        # self.episode_count += 1  # 这行实际上不需要，因为process_frame会处理
        
        # 重置状态准备下一局
        self.last_frame = None
        self.last_observations = {}
        self.frame_count = 0
        self.last_tick = 0
    
    def print_progress(self):
        """打印当前进度（不保存）"""
        if len(self.collectors) == 0:
            return
            
        total_transitions = sum(c.stats['total_transitions'] for c in self.collectors.values())
        print(f"📈 进度 - Episode {self.episode_count}, "
              f"Frame {self.frame_count}, "
              f"Tick {self.last_tick}, "
              f"Transitions {total_transitions}")
    
    def save_all(self, episode_num: int):
        """保存所有专家的数据"""
        saved_files = []
        for player_id, collector in self.collectors.items():
            filename = f"expert_{player_id}_ep{episode_num:04d}.pkl"  # 4位数字编号
            filepath = collector.save_data(filename)
            saved_files.append(filepath)
            collector.clear_buffer()
        
        print(f"\n✅ Episode {episode_num} 所有专家数据已保存")
        for filepath in saved_files:
            print(f"   📁 {filepath}")
    
    def print_statistics(self):
        """打印统计信息"""
        print("\n" + "="*60)
        print("实时专家数据收集统计")
        print("="*60)
        print(f"总Episodes: {self.episode_count}")
        print(f"当前Episode帧数: {self.frame_count}")
        print(f"收集的专家数: {len(self.collectors)}")
        
        for player_id, collector in self.collectors.items():
            print(f"\n专家 {player_id}:")
            print(f"  Episodes: {collector.stats['total_episodes']}")
            print(f"  Transitions: {collector.stats['total_transitions']}")
            
            # 打印MultiDiscrete动作分布
            if "direction" in collector.stats['action_distribution']:
                direction_names = ["不动", "上", "下", "左", "右"]
                print(f"  方向分布:")
                for i, count in collector.stats['action_distribution']['direction'].items():
                    if count > 0:
                        print(f"    {direction_names[i]}: {count}")
                
                print(f"  炸弹分布:")
                for i, count in collector.stats['action_distribution']['bomb'].items():
                    bomb_name = "不放" if i == 0 else "放"
                    print(f"    {bomb_name}: {count}")
                
                speed_names = ["最大", "极慢", "慢", "中", "快"]
                print(f"  速度分布:")
                for i, count in collector.stats['action_distribution']['speed'].items():
                    if count > 0:
                        print(f"    {speed_names[i]}: {count}")
        print("="*60)


# 全局实例（在robot.py中使用）
_global_collector: Optional[RealtimeExpertCollector] = None


def get_global_collector(save_dir: str = "./expert_data") -> RealtimeExpertCollector:
    """获取全局收集器实例"""
    global _global_collector
    if _global_collector is None:
        _global_collector = RealtimeExpertCollector(save_dir=save_dir)
    return _global_collector


def enable_data_collection(save_dir: str = "./expert_data", auto_save: bool = True, max_ticks: int = 1800):
    """
    启用数据收集
    
    Args:
        save_dir: 数据保存目录
        auto_save: 是否自动检测游戏结束并保存
        max_ticks: 游戏最大tick数
    """
    global _global_collector
    _global_collector = RealtimeExpertCollector(
        save_dir=save_dir, 
        auto_save=auto_save, 
        max_ticks=max_ticks
    )
    return _global_collector


def disable_data_collection():
    """禁用数据收集"""
    global _global_collector
    if _global_collector is not None:
        _global_collector.print_statistics()
        _global_collector = None
    print("❌ 实时数据收集已禁用")


def is_collection_enabled() -> bool:
    """检查数据收集是否启用"""
    return _global_collector is not None

