# This patch tricks libraries like stable-baselines3 that still import "gym"
# into importing "gymnasium" instead. This is the standard way to deal with
# the deprecation warning and must be done before those libraries are imported.
import gymnasium
import sys
sys.modules['gym'] = gymnasium

import time
import threading
import queue
import functools
from flask import Flask, request, jsonify

# --- Data models and core imports ---
from data_models import Frame
from core_model import preprocess_observation_dict

# --- Refactored Module Imports ---
# Each module now has a specific responsibility.
from agent_manager import get_or_create_agent_state
from training_worker import training_worker
import strategy
from shared_state import load_or_create_model

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Global Instances (created in main) ---
# These will be created once and injected into the functions that need them.
main_model = None
main_trajectory_queue = None

# --- API Endpoint for Agent Commands ---
@app.route("/api/v1/command", methods=["POST"])
def handle_command():
    """
    This is the main endpoint that receives game state and returns an action.
    It orchestrates the process of decision-making and experience gathering.
    """
    start_time = time.time()
    
    data = request.get_json()
    current_frame = Frame(data)
    robot_id = current_frame.my_player.id
    
    state = get_or_create_agent_state(robot_id)
    
    processed_observation = preprocess_observation_dict(current_frame)
    state['raw_frame_buffer'].append(current_frame)
    state['processed_frame_buffer'].append(processed_observation)
    
    # Use the injected instances
    action_id, value, log_prob = strategy.decide_action(state, main_model)
    strategy.collect_experience(robot_id, state, current_frame, main_trajectory_queue)
    
    state['last_action'] = action_id
    state['last_value'] = value
    state['last_log_prob'] = log_prob
    
    response_data = strategy.action_to_response(action_id)
    
    end_time = time.time()
    elapsed_ms = (end_time - start_time) * 1000
    #print(f"--- Request for robot {robot_id} handled in {elapsed_ms:.2f} ms. Action: {action_id} ---")
    
    return jsonify(response_data)

# --- Health Check Endpoint ---
@app.route("/api/v1/ping", methods=["HEAD"])
def handle_ping():
    """
    A simple endpoint to check if the server is running.
    """
    return "", 200

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Dependency Injection Setup ---
    # 1. Create the shared instances here in the main entry point.
    main_model = load_or_create_model()
    main_trajectory_queue = queue.Queue()

    # 2. Create partial functions with the instances "injected".
    #    (We don't need to do this for the handle_command function as it can access the globals directly)
    injected_training_worker = functools.partial(training_worker, model=main_model, trajectory_queue=main_trajectory_queue)

    # 3. Start the background thread with the injected function.
    print("Starting the training worker thread...")
    trainer_thread = threading.Thread(target=injected_training_worker, daemon=False)
    trainer_thread.start()
    
    # Suppress Flask's default POST request logging.
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.WARNING)

    # Start the Flask web server.
    print("Starting Flask server...")
    app.run(host="0.0.0.0", port=5002, debug=False, threaded=True)
