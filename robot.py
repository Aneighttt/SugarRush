from flask import Flask, request, jsonify
import time
import atexit
import os
import threading
from stable_baselines3.common.env_util import make_vec_env
from frame_processor import preprocess_observation_dict
from data_models import Frame
from agent import SB3_DQNAgent
from environment import BomberEnv
import gymnasium as gym
import sys
import random
sys.modules["gym"] =gym
app = Flask(__name__)

# Suppress the default Flask request logs
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# --- Environment and Agent Management ---
# We use a VecEnv to handle multiple environments (one for each player).
# n_envs should match the number of players you expect to control.
vec_env = make_vec_env(BomberEnv, n_envs=2)

# A single, shared agent learns from the vectorized environment.
shared_agent = SB3_DQNAgent(env=vec_env)
try:
    shared_agent.load("bomberman_dqn_2v2.zip")
    print("--- Shared agent model weights loaded successfully. ---")
except FileNotFoundError:
    print("--- No pre-trained model found for shared agent, starting from scratch. ---")

# --- State Tracking ---
player_to_env_map = {}
next_env_idx = 0
global_lock = threading.Lock()
from utils import *
from stable_baselines3.common.callbacks import BaseCallback

class PrintCallback(BaseCallback):
    def __init__(self, check_freq=100, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # The exploration_rate is a property of the DQN model itself
            exploration_rate = self.model.exploration_rate
            print(f"[Step {self.num_timesteps}] Exploration Rate: {exploration_rate:.4f}")
        
        # Also, print episode summary when it ends
        if "rollout/ep_rew_mean" in self.model.logger.name_to_value:
            ep_rew_mean = self.model.logger.get_mean("rollout/ep_rew_mean")
            print(f"\n--- Episode End ---")
            print(f"  Steps: {self.num_timesteps}")
            print(f"  Mean Reward: {ep_rew_mean:.2f}")
            print(f"-------------------\n")
        return True


def training_loop():
    """
    The main training loop. It runs in a single background thread.
    """
    print("--- Starting training thread ---")
    # We check every 100 steps. log_interval=1 ensures episode summaries are logged.
    shared_agent.model.learn(total_timesteps=int(1e7), log_interval=1, callback=PrintCallback(check_freq=1))

training_thread = threading.Thread(target=training_loop, daemon=True)
training_thread.start()

@app.route("/api/v1/command", methods=["POST"])
def handle_command():
    start_time = time.time()
    
    global next_env_idx
    data = request.get_json()
    frame = Frame(data)
    player_id = frame.my_player.id

    # --- Player to Environment Mapping ---
    if player_id not in player_to_env_map:
        with global_lock:
            if player_id not in player_to_env_map:
                if next_env_idx >= vec_env.num_envs:
                    return jsonify({"error": "Max number of players reached"}), 500
                player_to_env_map[player_id] = next_env_idx
                next_env_idx += 1
    env_idx = player_to_env_map[player_id]

    processed_frame = preprocess_observation_dict(frame)
    if(player_id==3):
        print_observation(processed_frame, 2)
        print_pixel_view_terminal(processed_frame["pixel_view"], downsample_factor=2)
    #vec_env.env_method('put_frame', frame, indices=[env_idx])
    after_put_frame_time = time.time()

    # 2. Get the action from the correct sub-environment's process (this is a blocking call)
    #action = vec_env.env_method('get_action', indices=[env_idx])[0]
    after_get_action_time = time.time()

    # --- Translate action to game command ---
    action = 0
    direction = "N"
    is_place_bomb = False
    if action == 0: direction = "U"
    elif action == 1: direction = "D"
    elif action == 2: direction = "L"
    elif action == 3: direction = "R"
    elif action == 4:
        my_bombs_on_map = sum(1 for bomb in frame.bombs if bomb.owner_id == frame.my_player.id)
        if my_bombs_on_map < frame.my_player.bomb_pack_count:
            is_place_bomb = True
    
    speed = 10 + frame.my_player.agility_boots_count * 2
    direction = random.choice(["U", "D", "L", "R", "N"])
    is_place_bomb = random.choice([True, False])

    response_data = {
        "direction": direction,
        "is_place_bomb": is_place_bomb,
        "stride": speed
    }
    
    end_time = time.time()

    # --- Detailed Timing Logs ---
    put_frame_duration = (after_put_frame_time - start_time) * 1000
    get_action_duration = (after_get_action_time - after_put_frame_time) * 1000
    response_prep_duration = (end_time - after_get_action_time) * 1000
    total_duration = (end_time - start_time) * 1000
    
    # print(
    #     f"[Player {player_id} | Env {env_idx}] "
    #     f"Total: {total_duration:.2f}ms | "
    #     f"PutFrame: {put_frame_duration:.2f}ms | "
    #     f"GetAction: {get_action_duration:.2f}ms | "
    #     f"ResponsePrep: {response_prep_duration:.2f}ms"
    # )
    
    return jsonify(response_data)

@app.route("/api/v1/ping", methods=["HEAD"])
def handle_ping():
    return "", 200

def on_exit():
    print(f"--- ON_EXIT: Saving shared agent model. ---")
    shared_agent.save("bomberman_dqn_2v2.zip")

atexit.register(on_exit)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=False, threaded=False)
