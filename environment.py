import gymnasium as gym
from gymnasium import spaces
import numpy as np
import queue
from collections import deque
import os

from data_models import Frame
from frame_processor import preprocess_observation_dict
from reward import calculate_reward, count_territory

# --- Debugging Helper ---
def print_observation(obs, header=""):
    """Formats and prints the observation dictionary for debugging."""
    np.set_printoptions(precision=2, suppress=True, linewidth=120)
    print(f"\n--- {header} (PID: {os.getpid()}) ---")
    
    # Print Player State
    player_state = obs["player_state"]
    print(f"Player State: {player_state}")

    # Print Grid View (current frame's channels)
    grid_view = obs["grid_view"]
    current_grid_view = grid_view[12:, :, :] # Get the last 12 channels
    
    channel_names = [
        "0: Terrain", "1: Bombs", "2: Danger Zone", "3: Item (Boots)",
        "4: Item (Potion)", "5: Item (Bomb Pack)", "6: Accel Terrain",
        "7: Decel Terrain", "8: Enemy Territory", "9: My Territory",
        "10: My Player Pos", "11: Teammate Pos" # Assuming these are the last two channels
    ]
    
    print("--- Grid View (Current Frame) ---")
    for i, name in enumerate(channel_names):
        channel = current_grid_view[i, :, :]
        # Only print if the channel contains non-zero values
        if np.any(channel):
            print(f"Channel {name}:")
            print(channel)

    print("-----------------------------------\n")


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
            "grid_view": spaces.Box(low=0, high=1, shape=(12 * 2, 11, 11), dtype=np.float32),
            "pixel_view": spaces.Box(low=0, high=1, shape=(2 * 2, 100, 100), dtype=np.float32),
            "player_state": spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
        })

        self.frame_queue = queue.Queue()
        self.action_queue = queue.Queue()
        self.observation_history = deque(maxlen=2)
        self.raw_frame_history = deque(maxlen=2)
        self.previous_frame_info = {}

    def _get_stacked_observation(self):
        prev_obs = self.observation_history[0]
        curr_obs = self.observation_history[-1]
        
        # Manually crop the 16x28 grid_view to an 11x11 view
        # Assuming the center of the 16x28 view is where the player is.
        # This is a placeholder logic, actual cropping might need adjustment.
        center_h, center_w = 16 // 2, 28 // 2
        half_view = 11 // 2
        
        def crop_view(grid):
            return grid[:, center_h-half_view:center_h+half_view+1, center_w-half_view:center_w+half_view+1]

        cropped_prev_grid = crop_view(prev_obs["grid_view"])
        cropped_curr_grid = crop_view(curr_obs["grid_view"])

        return {
            "grid_view": np.concatenate([cropped_prev_grid, cropped_curr_grid], axis=0),
            "pixel_view": np.concatenate([prev_obs["pixel_view"], curr_obs["pixel_view"]], axis=0),
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
        print_observation(stacked_observation, header=f"RESET - Player {initial_frame.my_player.id}")
        return stacked_observation, {}

    def step(self, action: int):
        self.action_queue.put(action)
        current_frame = self.frame_queue.get()
        new_observation = preprocess_observation_dict(current_frame)
        self.observation_history.append(new_observation)
        self.raw_frame_history.append(current_frame)
        previous_frame = self.raw_frame_history[0]
        reward, new_frame_info = calculate_reward(current_frame, previous_frame, self.previous_frame_info, action)
        self.previous_frame_info = new_frame_info
        done = current_frame.current_tick == 1800
        stacked_observation = self._get_stacked_observation()
        print_observation(stacked_observation, header=f"STEP - Player {current_frame.my_player.id} Tick {current_frame.current_tick}")
        return stacked_observation, reward, done, False, {}

    def put_frame(self, frame):
        self.frame_queue.put(frame)

    def get_action(self):
        return self.action_queue.get()
