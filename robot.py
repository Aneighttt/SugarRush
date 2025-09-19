from flask import Flask, request, jsonify
import json
import time
from data_models import Frame

app = Flask(__name__)


@app.route("/api/v1/command", methods=["POST"])
def handle_command():
    start_time = time.time()

    data = request.get_json()
    frame = Frame(data)
    
    response_data = {
        "direction": "N", # N 代表无方向/静止
        "is_place_bomb": False,
        "stride": 0
    }
    
    end_time = time.time()
    elapsed_ms = (end_time - start_time) * 1000
    print(f"--- Request handled in {elapsed_ms:.2f} ms ---")
    
    return jsonify(response_data)

@app.route("/api/v1/ping", methods=["HEAD"])
def handle_ping():
    return "", 200

if __name__ == "__main__":
    # 使用 debug=False 和 threaded=False 以获得更稳定的行为
    app.run(host="0.0.0.0", port=5002, debug=False, threaded=False)
