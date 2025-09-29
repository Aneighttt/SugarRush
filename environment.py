import gymnasium as gym
from gymnasium import spaces
import numpy as np
import queue
from collections import deque
import os
import time

from data_models import Frame
from frame_processor import preprocess_observation_dict
from reward import calculate_reward, count_territory


class BomberEnv(gym.Env):
    """
    A custom Gym environment for the Bomberman game that interacts with an
    external game server via queues. This is designed to be run in a
    separate process as part of a VecEnv.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(BomberEnv, self).__init__()

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Dict({
            "grid_view": spaces.Box(low=0, high=1, shape=(10 * 2, 11, 11), dtype=np.float32),
            "pixel_view": spaces.Box(low=0, high=1, shape=(2 * 2, 100, 100), dtype=np.float32),
            "player_state": spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
        })

        self.frame_queue = queue.Queue()
        self.action_queue = queue.Queue()
        self.observation_history = deque(maxlen=2)
        self.raw_frame_history = deque(maxlen=2)
        self.previous_frame_info = {}

    def _get_stacked_observation(self):
        """Stacks the grid and pixel views from the last two observations."""
        prev_obs = self.observation_history[0]
        curr_obs = self.observation_history[-1]

        # The grid_view is already an 11x11 view centered on the player.
        # No cropping is needed here.
        stacked_grid_view = np.concatenate([prev_obs["grid_view"], curr_obs["grid_view"]], axis=0)
        stacked_pixel_view = np.concatenate([prev_obs["pixel_view"], curr_obs["pixel_view"]], axis=0)

        return {
            "grid_view": stacked_grid_view,
            "pixel_view": stacked_pixel_view,
            "player_state": curr_obs["player_state"]
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        initial_frame = self.frame_queue.get()
        initial_observation = preprocess_observation_dict(initial_frame)
        self.observation_history.clear()
        self.raw_frame_history.clear()
        self.observation_history.append(initial_observation)
        self.observation_history.append(initial_observation)
        self.raw_frame_history.append(initial_frame)
        self.raw_frame_history.append(initial_frame)
        my_territory, enemy_territory = count_territory(initial_frame)
        self.previous_frame_info = {
            'my_territory': my_territory, 'enemy_territory': enemy_territory,
            'is_stunned': (initial_frame.my_player.status == 'D'),
            'items_collected': initial_frame.my_player.bomb_pack_count + initial_frame.my_player.sweet_potion_count + initial_frame.my_player.agility_boots_count,
            'my_bomb_identifiers': {(b.position.x, b.position.y, b.explode_at) for b in initial_frame.bombs if b.owner_id == initial_frame.my_player.id},
            'last_action': -1
        }
        stacked_observation = self._get_stacked_observation()
        #print_observation(stacked_observation, header=f"RESET - Player {initial_frame.my_player.id}")
        return stacked_observation, {}

    def step(self, action: int):
        step_start_time = time.time()
        self.action_queue.put(action)
        
        # This call blocks until a frame is put into the queue by the web server thread
        current_frame = self.frame_queue.get()
        after_get_frame_time = time.time()

        new_observation = preprocess_observation_dict(current_frame)
        self.observation_history.append(new_observation)
        self.raw_frame_history.append(current_frame)
        previous_frame = self.raw_frame_history[0]
        reward, new_frame_info = calculate_reward(current_frame, previous_frame, self.previous_frame_info, action)
        self.previous_frame_info = new_frame_info
        done = current_frame.current_tick == 1800
        stacked_observation = self._get_stacked_observation()
        #print_observation(stacked_observation, header=f"STEP - Player {current_frame.my_player.id} Tick {current_frame.current_tick}")
        
        step_end_time = time.time()

        # --- Detailed Step Timing Logs ---
        wait_for_frame_duration = (after_get_frame_time - step_start_time) * 1000
        processing_duration = (step_end_time - after_get_frame_time) * 1000
        total_step_duration = (step_end_time - step_start_time) * 1000
        
        print(
            f"[Env Step | Player {current_frame.my_player.id}] "
            f"Total: {total_step_duration:.2f}ms | "
            f"WaitingForFrame: {wait_for_frame_duration:.2f}ms | "
            f"Processing: {processing_duration:.2f}ms"
        )

        return stacked_observation, reward, done, False, {}

    def put_frame(self, frame):
        self.frame_queue.put(frame)

    def get_action(self):
        return self.action_queue.get()
