import json
import os
from flask import Flask, jsonify, render_template

app = Flask(__name__)

LOG_DIR = "logs"
# To accommodate different player IDs that the server might assign,
# we now monitor all potential player IDs.
PLAYER_IDS = ["1", "2", "3", "4"]

@app.route('/')
def index():
    """Serves the main visualization dashboard."""
    return render_template('monitor.html', player_ids=PLAYER_IDS)

@app.route('/data/<player_id>')
def get_player_data(player_id):
    """Provides the latest visualization data for a specific player."""
    filepath = os.path.join(LOG_DIR, f"viz_{player_id}.json")
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
            # --- DEBUG LOG ---
            print(f"[SUCCESS] Reading data for player {player_id}: {data}")
            return jsonify(data)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        # --- DEBUG LOG ---
        print(f"[ERROR] Could not read data for player {player_id} from {filepath}. Reason: {e}")
        return jsonify({"error": "Data not available yet."}), 404

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5003, debug=True)
