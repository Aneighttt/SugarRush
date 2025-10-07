from flask import Flask, request, jsonify
import random

app = Flask(__name__)

@app.route("/api/v1/command", methods=["POST"])
def handle_command():
    data = request.get_json()
    print(f"Tick: {data.get('current_tick')}")
    direction = random.choice(["U", "D", "L", "R", "N"])
    place_bomb = random.choice([True, False])
    response_data = {
        "direction": direction,
        "is_place_bomb": False,
    }
    return jsonify(response_data)

@app.route("/api/v1/ping", methods=["HEAD"])
def handle_ping():
	return "", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003)