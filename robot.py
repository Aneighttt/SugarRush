from flask import Flask, request, jsonify
import json
import time
from data_models import Frame
from strategy import AIStrategy

app = Flask(__name__)

# --- AI 初始化 ---
# 地图尺寸: 28 * 50 = 1400px, 16 * 50 = 800px
# 格子大小: 50px
MAP_WIDTH_PX = 28 * 50
MAP_HEIGHT_PX = 16 * 50
GRID_SIZE = 50

# 创建AI策略实例
ai_strategy = AIStrategy(MAP_WIDTH_PX, MAP_HEIGHT_PX, GRID_SIZE)
# -----------------

@app.route("/api/v1/command", methods=["POST"])
def handle_command():
    start_time = time.time()

    data = request.get_json()
    frame = Frame(data)
    
    # 调用AI大脑进行决策
    action_command = ai_strategy.make_decision(frame)
    
    # 如果AI给出了指令，就使用它
    if action_command:
        response_data = action_command
    else:
        # 否则，原地不动
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
