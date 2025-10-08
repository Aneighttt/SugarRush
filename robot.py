from flask import Flask, request, jsonify
import time
import atexit
import os
import threading
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from frame_processor import preprocess_observation_dict
from data_models import Frame
from agent import SB3_DQNAgent
from environment import BomberEnv
from utils import calculate_distance_map_to_frontier
import gymnasium as gym
import sys
from filelock import FileLock
import numpy as np
import collections
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.logger import configure

sys.modules["gym"] = gym
app = Flask(__name__)

# Suppress the default Flask request logs
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# --- Constants and Global Objects ---
MODEL_PATH = "./model/bomberman_dqn_2v2.zip"
LOCK_PATH = f"{MODEL_PATH}.lock"
PURE_PREDICTION_MODE = False # Master switch to disable training and run only prediction.
model_lock = FileLock(LOCK_PATH)

# --- Environment and Agent Management ---
# Vectorized environment for training
n_envs = 2


player_raw_frame = collections.defaultdict(lambda: collections.deque(maxlen=2))
player_processed_frame = collections.defaultdict(lambda: collections.deque(maxlen=2))
player_action = collections.defaultdict(lambda: collections.deque(maxlen=2))

# --- Per-Player Caches for Pathfinding ---
player_cached_distance_map = collections.defaultdict(lambda: None)
player_last_map_calculation_tick = collections.defaultdict(lambda: -100)

# A single, shared agent for trainingewa
if not PURE_PREDICTION_MODE:
    train_vec_env = make_vec_env(BomberEnv, n_envs=n_envs)
    training_agent = SB3_DQNAgent(env=train_vec_env)
    try:
        with model_lock:
            if os.path.exists(MODEL_PATH):
                training_agent.load(MODEL_PATH,fine_tuning=False)
                print("--- Training agent model weights loaded successfully. ---")
    except FileNotFoundError:
        print("--- No pre-trained model found for training agent, starting from scratch. ---")

# A separate agent for prediction
# We only need a single environment for the prediction agent's structure
prediction_env = BomberEnv()
prediction_agent = SB3_DQNAgent(env=prediction_env)
try:
    with model_lock:
        if os.path.exists(MODEL_PATH):
            prediction_agent.load(MODEL_PATH, load_replay_buffer=False)
            print("--- Prediction agent model weights loaded successfully. ---")
except FileNotFoundError:
    print("--- No pre-trained model found for prediction agent, using initial model. ---")


# --- State Tracking ---
player_to_env_map = {}
next_env_idx = 0
global_lock = threading.Lock()

# --- Path for the final model of the current run ---
final_model_save_path = MODEL_PATH

# --- Custom Callback for Saving Model ---
class SaveModelCallback(BaseCallback):
    def __init__(self, save_freq, save_path, lock, is_checkpoint=False, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.lock = lock
        self.is_checkpoint = is_checkpoint

    def _on_step(self) -> bool:
        if self.n_calls > 0 and self.n_calls % self.save_freq == 0:
            path = self.save_path
            if self.is_checkpoint:
                # For checkpoints, save directly into the run-specific directory
                path = os.path.join(self.save_path, f"model_{self.num_timesteps}_steps.zip")
                print(f"--- Saving model checkpoint to {path} ---")
            else:
                print(f"--- Saving main model to {path} ---")

            with self.lock:
                # For checkpoints, only save the model file, not the replay buffer.
                self.model.save(path)
            
            if self.is_checkpoint:
                print(f"--- Model checkpoint saved successfully (model only). ---")
            else:
                print(f"--- Main model saved successfully. ---")
        return True

class TrainLogCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TrainLogCallback, self).__init__(verbose)
        self.ep_reward = 0
        self.ep_len = 0
        
        # --- CSV Logging Setup ---
        self.log_dir = "reward"
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, "reward_log.csv")
        
        # Define the header based on the keys in reward.py, ensuring it's up-to-date
        self.all_reward_keys = [
            'territory_diff', 'capture_tile', 'win', 'lose', 'stun', 'item_collect',
            'bomb_strategy', 'bomb_limit_penalty', 'move_reward', 'not_moving', 
            'do_nothing', 'living_penalty', 'enter_danger_zone', 'exit_danger_zone',
            'staying_in_danger', 'moved_closer_to_safety', 'follow_gradient_path'
        ]
        header = "step," + ",".join(self.all_reward_keys) + ",total_reward\n"
        
        # Open file and write header
        self.reward_log_file = open(self.log_path, "w")
        self.reward_log_file.write(header)
        self.reward_log_file.flush()

    def _on_step(self) -> bool:
        # --- TensorBoard Logging (existing logic) ---
        reward = self.locals["rewards"][0]
        self.logger.record("train/step_reward", reward)
        self.ep_reward += reward
        self.ep_len += 1
        if self.locals["dones"][0]:
            self.logger.record("train/episode_reward", self.ep_reward)
            self.logger.record("train/episode_length", self.ep_len)
            self.ep_reward = 0
            self.ep_len = 0
        self.logger.dump(step=self.num_timesteps)

        # --- CSV Logging (new logic) ---
        infos = self.locals.get("infos", [{}])
        if "reward_dict" in infos[0]:
            reward_dict = infos[0]["reward_dict"]
            total_reward = sum(reward_dict.values())
            
            log_values = [str(self.num_timesteps)] + \
                         [str(reward_dict.get(key, 0.0)) for key in self.all_reward_keys] + \
                         [str(total_reward)]
            
            self.reward_log_file.write(",".join(log_values) + "\n")
            self.reward_log_file.flush()

        return True

    def close(self):
        if hasattr(self, 'reward_log_file') and not self.reward_log_file.closed:
            self.reward_log_file.close()
            print("--- Reward log file closed. ---")

# --- Training and Prediction Threads ---
def training_loop():
    """
    The main training loop. It runs in a single background thread.
    """
    print("--- Starting training thread ---")

    # --- Setup for Checkpoint Saving ---
    run_name = f"run_{time.strftime('%Y%m%d-%H%M%S')}"
    checkpoint_dir = os.path.join("model", run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"--- Checkpoints for this run will be saved in: {checkpoint_dir} ---")
    
    # Set the global path for the final model save (still uses the old logic for on_exit)
    global final_model_save_path
    final_model_save_path = MODEL_PATH

    # --- Callbacks ---
    global train_log_callback_instance
    # 1. Main model callback (saves to root model dir every ~10 seconds, assuming ~30 steps/sec)
    main_model_callback = SaveModelCallback(save_freq=300, save_path=MODEL_PATH, lock=model_lock)
    
    # 2. Checkpoint callback (saves to run-specific dir every 10k steps)
    checkpoint_callback = SaveModelCallback(save_freq=10000, save_path=checkpoint_dir, lock=model_lock, is_checkpoint=True)

    # 3. Logging callback
    train_log_callback_instance = TrainLogCallback()

    callback = CallbackList([main_model_callback, checkpoint_callback, train_log_callback_instance])
    
    # Use the unique run_name for TensorBoard logs
    training_agent.model.learn(total_timesteps=720_000, tb_log_name=run_name, callback=callback)

def prediction_model_updater_loop():
    """
    This loop periodically checks for and loads the latest saved model for prediction.
    """
    print("--- Starting prediction model update thread ---")
    while True:
        # Check for a new model every 15 seconds
        time.sleep(15)
        if os.path.exists(MODEL_PATH):
            try:
                with model_lock:
                    print("--- Loading latest model for prediction ---")
                    prediction_agent.load(MODEL_PATH, load_replay_buffer=False)
                    print("--- Prediction model updated successfully ---")
            except Exception as e:
                print(f"Could not load prediction model: {e}")

if not PURE_PREDICTION_MODE:
    training_thread = threading.Thread(target=training_loop, daemon=True)
    training_thread.start()

    prediction_thread = threading.Thread(target=prediction_model_updater_loop, daemon=True)
    prediction_thread.start()
else:
    print("\n--- RUNNING IN PURE PREDICTION MODE - TRAINING IS DISABLED ---\n")


@app.route("/api/v1/command", methods=["POST"])
def handle_command():
    timings = {}
    start_time = time.time()
    last_time = time.time()

    global next_env_idx
    data = request.get_json()
    frame = Frame(data)
    player_id = frame.my_player.id

    timings["get_data"] = (time.time() - last_time) * 1000
    last_time = time.time()

    # --- Pathfinding Calculation with Per-Player Caching based on Ticks ---
    # This is thread-safe because each player (and thus each thread)
    # has its own entry in the defaultdict.
    if frame.current_tick - player_last_map_calculation_tick[player_id] >= 10 or player_cached_distance_map[player_id] is None:
        player_cached_distance_map[player_id] = calculate_distance_map_to_frontier(frame)
        player_last_map_calculation_tick[player_id] = frame.current_tick
    
    # Attach the map to the frame object for downstream use
    frame.distance_map = player_cached_distance_map[player_id]
    # Do the same for the previous frame if it exists
    if len(player_raw_frame[player_id]) > 0:
        prev_frame = player_raw_frame[player_id][0]
        # Attach the most recent map, as the previous frame wouldn't have triggered a calc.
        prev_frame.distance_map = player_cached_distance_map[player_id]

    timings["pathfinding"] = (time.time() - last_time) * 1000
    last_time = time.time()

    # --- Player to Environment Mapping for Training ---
    if not PURE_PREDICTION_MODE:
        if player_id not in player_to_env_map:
            with global_lock:
                if player_id not in player_to_env_map:
                    if next_env_idx >= train_vec_env.num_envs:
                        return jsonify({"error": "Max number of players reached"}), 500
                    player_to_env_map[player_id] = next_env_idx
                    next_env_idx += 1
        env_idx = player_to_env_map[player_id]

    timings["player_mapping"] = (time.time() - last_time) * 1000
    last_time = time.time()
    
    # --- Preprocess Frame and Update Training Environment ---
    processed_frame = preprocess_observation_dict(frame)

    timings["preprocess"] = (time.time() - last_time) * 1000
    last_time = time.time()

    player_raw_frame[frame.my_player.id].append(frame)
    player_processed_frame[frame.my_player.id].append(processed_frame)

    # Use the current processed frame directly for decision making
    current_obs = player_processed_frame[player_id][-1]
    
    action = prediction_agent.choose_action(current_obs, deterministic=PURE_PREDICTION_MODE)
    player_action[frame.my_player.id].append(action)

    timings["choose_action"] = (time.time() - last_time) * 1000
    last_time = time.time()

    # Pass the previous and current frames (both raw and processed) to the environment
    if not PURE_PREDICTION_MODE and len(player_raw_frame[player_id]) > 1:
        train_vec_env.env_method(
            'put_frame_pair',
            player_processed_frame[player_id][0],  # Previous processed obs (P_T-1)
            player_processed_frame[player_id][1],  # Current processed obs (P_T)
            player_raw_frame[player_id][0],        # Previous raw frame (T-1)
            player_raw_frame[player_id][1],        # Current raw frame (T)
            player_action[player_id][0],           # Action taken at previous obs (A_T-1)
            indices=[env_idx]
        )
    
    timings["update_env"] = (time.time() - last_time) * 1000
    last_time = time.time()

    # --- Translate action to game command ---
    # Action space: 0=Up, 1=Down, 2=Left, 3=Right, 4=Bomb, 5=Stay
    direction = "N"
    is_place_bomb = False
    if action == 0:
        direction = "U"
    elif action == 1:
        direction = "D"
    elif action == 2:
        direction = "L"
    elif action == 3:
        direction = "R"
    elif action == 4:
        # The check for whether a bomb can be placed is handled by the game server.
        # The agent should be free to attempt the action.
        is_place_bomb = True
    elif action == 5:
        direction = "N" # Explicitly stay
    
    speed = 0

    response_data = {
        "direction": direction,
        "is_place_bomb": is_place_bomb,
        "stride": speed
    }

    timings["translate_action"] = (time.time() - last_time) * 1000
    timings["total_time"] = (time.time() - start_time) * 1000
    if timings["total_time"] >= 85:
        print(f"Tick {frame.current_tick}: { {k: f'{v:.2f}ms' for k, v in timings.items()} }")
    #print(f"Response data: {response_data}")
    return jsonify(response_data)

@app.route("/api/v1/ping", methods=["HEAD"])
def handle_ping():
    return "", 200

train_log_callback_instance = None

def on_exit():
    if not PURE_PREDICTION_MODE:
        print(f"--- ON_EXIT: Saving final training agent model. ---")
        with model_lock:
            # Save to the main model path on exit
            training_agent.save(MODEL_PATH)
        print(f"--- Final model saved to {MODEL_PATH}. ---")
    
    # Close the log file
    if train_log_callback_instance:
        train_log_callback_instance.close()

atexit.register(on_exit)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=False, threaded=True)
