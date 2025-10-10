import gymnasium as gym
from gymnasium import spaces
import numpy as np
import queue
from collections import deque
import os
import time
import logging
from torch.utils.tensorboard import SummaryWriter
from data_models import Frame
from frame_processor import preprocess_observation_dict
from reward import calculate_reward, count_territory


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class BomberEnv(gym.Env):
    """
    A custom Gym environment for the Bomberman game that interacts with an
    external game server via queues. This is designed to be run in a
    separate process as part of a VecEnv.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(BomberEnv, self).__init__()

        # 新的动作空间：MultiDiscrete [方向(5), 放炸弹(2), 速度(5)]
        # 方向: 0=不动, 1=上, 2=下, 3=左, 4=右
        # 放炸弹: 0=不放, 1=放
        # 速度: 0=最大(16), 1=极慢(2), 2=慢(4), 3=中(8), 4=快(12)
        self.action_space = spaces.MultiDiscrete([5, 2, 5])
        
        # grid_view shape: (channels, height, width) = (14, 16, 28)
        # 地图是28宽x16高，numpy中是(height, width)
        self.observation_space = spaces.Dict({
            "grid_view": spaces.Box(low=0, high=1, shape=(14, 16, 28), dtype=np.float32),
            "player_state": spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)  # 更新为10维
        })

        self.frame_queue = queue.Queue(maxsize=1000)
        # self.writer = SummaryWriter(log_dir="runs/reward_logs")
        self.num_steps = 0
        self.all_reward_keys = [
            'territory_diff', 'capture_tile', 'win', 'lose', 'stun', 'item_collect',
            'bomb_strategy', 'bomb_limit_penalty', 'move_reward', 'not_moving', 
            'do_nothing', 'living_penalty', 'enter_danger_zone', 'exit_danger_zone',
            'staying_in_danger', 'moved_closer_to_safety', 'follow_gradient_path'
        ]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        logging.info("Resetting environment and clearing frame queue.")
        
        # Clear any stale frames from the previous episode
        if not self.frame_queue.empty():
            l = self.frame_queue.qsize()
            print("{}".format(l))
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                continue
        
        # Block and wait for the first frame of the new episode
        obs, next_obs, raw_obs, next_raw_obs, real_action = self.frame_queue.get()
        
        # The first observation of a new episode is next_obs
        return next_obs, {}

    def step(self, action: int):
        step_start_time = time.time()
        obs, next_obs, raw_obs, next_raw_obs, real_action = self.frame_queue.get()
        
        # Pass the previous processed observation (obs) to calculate_reward
        reward_dict = calculate_reward(raw_obs, next_raw_obs, real_action, obs)
        total_reward = sum(reward_dict.values())

        # Create a full reward dictionary for logging and printing
        full_reward_dict = {key: reward_dict.get(key, 0.0) for key in self.all_reward_keys}
        
        print(f"Step {self.num_steps}, TotalReward: {total_reward} Rewards: {full_reward_dict}")

        # if reward_dict:
        #     # logging.info(f"Reward details: {reward_dict}")
        #     for key, value in reward_dict.items():
        #         self.writer.add_scalar(f'Reward/{key}', value, self.num_steps)
        #     self.writer.add_scalar('Reward/Total', total_reward, self.num_steps)
        
        self.num_steps += 1
        
        terminated = (next_raw_obs.current_tick >= 1800)
        truncated = False

        return next_obs, total_reward, terminated, truncated,  {"real_action": real_action, "reward_dict": full_reward_dict}

    def put_frame_pair(self, obs, next_obs, raw_obs, next_raw_obs, real_action):
        # This method now receives the full transition for a single step.
        # obs: P_T-1
        # next_obs: P_T
        # raw_obs: Raw T-1
        # next_raw_obs: Raw T
        # real_action: A_T-1
        self.frame_queue.put((obs, next_obs, raw_obs, next_raw_obs, real_action))

    def close(self):
        # This method is kept for API consistency, but no longer needs to close a file.
        pass
