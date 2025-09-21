from flask import Flask, request, jsonify
import time
import atexit
import os
import json
from data_models import Frame
from ai_logic import GameAI
from agent import DQNAgent
from utils import MAP_WIDTH, MAP_HEIGHT

app = Flask(__name__)

# --- Shared, Persistent AI Brain (DQNAgent) ---
# This is the single, persistent brain that all player instances will use.
# It holds the neural network, memory, and learning state (like epsilon).
# The input is now a self-centered 11x11 view.
VIEW_SIZE = 11
STATE_CHANNELS = 11
ACTION_SIZE = 6
INPUT_SHAPE = (STATE_CHANNELS, VIEW_SIZE, VIEW_SIZE)
shared_agent = DQNAgent(state_size=INPUT_SHAPE, action_size=ACTION_SIZE)
try:
    shared_agent.load("bomberman_dqn_2v2.pth")
    print("--- Shared agent model weights loaded successfully. ---")
except FileNotFoundError:
    print("--- No pre-trained model found for shared agent, starting from scratch. ---")


# --- Per-Game Player Instance Management ---
# This dictionary will store temporary GameAI instances, one for each player ID.
# These instances are reset every game.
game_players = {}
# We track the last seen tick to detect when a new game starts.
current_game_tick = -1

@app.route("/api/v1/command", methods=["POST"])
def handle_command():
    """
    Handles the command request from the game server.
    It detects new games, manages temporary player instances,
    and returns the command from the shared AI brain.
    """
    global current_game_tick, game_players, shared_agent
    start_time = time.time()

    data = request.get_json()
    frame = Frame(data)
    player_id = frame.my_player.id

    # --- Game Reset Detection ---
    if frame.current_tick < 10 and current_game_tick > 1790:
        print(f"--- New game detected (tick reset from {current_game_tick} to {frame.current_tick}), resetting player instances. ---")
        # Save the persistent, shared agent's progress.
        shared_agent.save("bomberman_dqn_2v2.pth")
        # Clear the temporary player instances from the previous game.
        game_players.clear()
    
    current_game_tick = max(current_game_tick, frame.current_tick)

    # Get or create a temporary player instance for the specific player
    if player_id not in game_players:
        print(f"--- Creating new player instance for player {player_id} ---")
        # Inject the one and only shared_agent into the new player instance.
        game_players[player_id] = GameAI(agent=shared_agent)
    
    player_instance = game_players[player_id]
    
    # Get the command from the player instance, which uses the shared brain
    response_data, viz_data = player_instance.get_command(frame, player_id)
    
    # Write visualization data to a player-specific file
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(os.path.join(log_dir, f"viz_{player_id}.json"), "w") as f:
        json.dump(viz_data, f)

    end_time = time.time()
    elapsed_ms = (end_time - start_time) * 1000
    print(f"--- Request for player {player_id} handled in {elapsed_ms:.2f} ms ---")
    
    return jsonify(response_data)

@app.route("/api/v1/ping", methods=["HEAD"])
def handle_ping():
    """Handles the ping request from the game server for health checks."""
    return "", 200

def on_exit():
    """
    Function to be called on application exit to save the shared AI model.
    """
    print(f"--- ON_EXIT: Saving shared agent model. ---")
    shared_agent.save("bomberman_dqn_2v2.pth")

# Register the save function to be called on exit
atexit.register(on_exit)

if __name__ == "__main__":
    # Use debug=False and threaded=False for stable, predictable behavior
    app.run(host="0.0.0.0", port=5002, debug=False, threaded=False)
