from flask import Flask, request, jsonify
import time
import atexit

from data_models import Frame
from ai_logic import GameAI

app = Flask(__name__)

# --- Global AI Instance Initialization ---
# All AI logic is now encapsulated within the GameAI class.
game_ai = GameAI()

@app.route("/api/v1/command", methods=["POST"])
def handle_command():
    """
    Handles the command request from the game server.
    It receives a frame, passes it to the AI for a decision,
    and returns the AI's command.
    """
    start_time = time.time()

    data = request.get_json()
    frame = Frame(data)
    
    # Get the command from our AI logic handler
    response_data = game_ai.get_command(frame)
    
    end_time = time.time()
    elapsed_ms = (end_time - start_time) * 1000
    print(f"--- Request handled in {elapsed_ms:.2f} ms ---")
    
    return jsonify(response_data)

@app.route("/api/v1/ping", methods=["HEAD"])
def handle_ping():
    """Handles the ping request from the game server for health checks."""
    return "", 200

def on_exit():
    """Function to be called on application exit to save the AI model."""
    game_ai.save_model()

# Register the save function to be called on exit
atexit.register(on_exit)

if __name__ == "__main__":
    # Use debug=False and threaded=False for stable, predictable behavior
    app.run(host="0.0.0.0", port=5002, debug=False, threaded=False)
