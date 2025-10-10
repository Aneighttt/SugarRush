from flask import Flask, request, jsonify
import time
import atexit
import os
import numpy as np
from frame_processor import preprocess_observation_dict
from data_models import Frame
from utils import calculate_distance_map_to_frontier
import collections
from realtime_expert_collector import enable_data_collection
import random
from stable_baselines3 import PPO
from environment import BomberEnv
from gymnasium.wrappers import FlattenObservation

app = Flask(__name__)

# Suppress the default Flask request logs
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# --- Constants and Global Objects ---
# PPO模型路径（优先级：PPO微调模型 > 加权BC模型 > 原版BC模型 > 随机AI）
PPO_FINETUNED_PATH = "./ppo_models/ppo_finetuned.zip"       # PPO微调后的模型
PPO_BC_WEIGHTED_PATH = "./bc_models_weighted/bc_ppo_weighted.zip"  # 加权BC模型 (新)
PPO_BC_INIT_PATH = "./bc_models_imitation/bc_ppo_ready.zip"  # 原版BC模型
ENABLE_EXPERT_DATA_COLLECTION = False  # 数据收集已完成，现在使用训练好的模型
USE_PPO_MODEL = True  # 是否使用PPO模型

# --- Per-Player Caches for Pathfinding (Performance Optimization) ---
player_cached_distance_map = collections.defaultdict(lambda: None)
player_last_map_calculation_tick = collections.defaultdict(lambda: -100)

# --- 加载PPO模型 ---
ppo_model = None
ppo_env = None  # 用于PPO预测的环境

if USE_PPO_MODEL:
    try:
        # 创建flattened环境（PPO需要）
        base_env = BomberEnv()
        ppo_env = FlattenObservation(base_env)
        
        # 优先加载PPO微调模型，然后是加权BC，最后是原版BC
        model_path = None
        if os.path.exists(PPO_FINETUNED_PATH):
            model_path = PPO_FINETUNED_PATH
            model_type = "PPO微调模型"
        elif os.path.exists(PPO_BC_WEIGHTED_PATH):
            model_path = PPO_BC_WEIGHTED_PATH
            model_type = "加权BC模型 (提升炸弹&减速权重)"
        elif os.path.exists(PPO_BC_INIT_PATH):
            model_path = PPO_BC_INIT_PATH
            model_type = "原版BC模型"
        
        if model_path:
            ppo_model = PPO.load(model_path, env=ppo_env)
            print(f"✅ {model_type}加载成功: {model_path}")
            print(f"   观察空间: Flattened({ppo_env.observation_space.shape})")
            print(f"   动作空间: {ppo_env.action_space}")
        else:
            print(f"⚠️  未找到PPO模型文件")
            print(f"   尝试路径: {PPO_FINETUNED_PATH}")
            print(f"   尝试路径: {PPO_BC_INIT_PATH}")
            print("   将使用随机AI")
            
    except Exception as e:
        print(f"❌ PPO模型加载失败: {e}")
        print("   将使用随机AI")
        ppo_model = None
        import traceback
        traceback.print_exc()
else:
    print("--- 使用随机AI ---")

# --- 启用专家数据收集 ---
if ENABLE_EXPERT_DATA_COLLECTION:
    expert_collector = enable_data_collection(
        save_dir="./expert_data",
        auto_save=True,      # 自动保存
        max_ticks=1800       # 游戏最大tick
    )
    print("\n--- 专家数据收集已启用（自动模式）---")
    print("--- 游戏结束时会自动保存数据 ---\n")
else:
    expert_collector = None


@app.route("/api/v1/command", methods=["POST"])
def handle_command():
    """处理游戏帧并返回AI决策"""
    timings = {}
    start_time = time.time()
    last_time = time.time()

    data = request.get_json()
    frame = Frame(data)
    player_id = frame.my_player.id

    timings["parse_frame"] = (time.time() - last_time) * 1000
    last_time = time.time()
    
    # --- 收集专家数据 ---
    if ENABLE_EXPERT_DATA_COLLECTION and expert_collector is not None:
        try:
            expert_collector.process_frame(frame)
        except Exception as e:
            print(f"❌ 数据收集出错: {e}")
    
    timings["data_collection"] = (time.time() - last_time) * 1000
    last_time = time.time()

    # --- Pathfinding Calculation (Per-Player Cache) ---
    if frame.current_tick - player_last_map_calculation_tick[player_id] >= 10 or \
       player_cached_distance_map[player_id] is None:
        player_cached_distance_map[player_id] = calculate_distance_map_to_frontier(frame)
        player_last_map_calculation_tick[player_id] = frame.current_tick
    
    frame.distance_map = player_cached_distance_map[player_id]

    timings["pathfinding"] = (time.time() - last_time) * 1000
    last_time = time.time()
    
    # --- Preprocess Frame for Prediction ---
    processed_frame = preprocess_observation_dict(frame)

    timings["preprocess"] = (time.time() - last_time) * 1000
    last_time = time.time()
    
    # --- AI Decision Making ---
    if ppo_model is not None:
        # 使用PPO模型预测
        try:
            # 1. Flatten观察（PPO需要flattened observation）
            flattened_obs = np.concatenate([
                processed_frame['grid_view'].flatten(),
                processed_frame['player_state']
            ]).astype(np.float32)
            
            # 2. PPO预测
            action, _states = ppo_model.predict(flattened_obs, deterministic=False)
            
            # action是numpy数组 [direction, bomb, speed]
            direction_action = int(action[0])
            bomb_action = int(action[1])
            speed_action = int(action[2])
            
        except Exception as e:
            print(f"⚠️  PPO预测出错: {e}")
            import traceback
            traceback.print_exc()
            # 回退到随机AI
            direction_action = random.randint(0, 4)
            bomb_action = random.randint(0, 1)
            speed_action = 0
    else:
        # 随机AI（用于测试或没有模型时）
        direction_action = random.randint(0, 4)
        bomb_action = random.randint(0, 1)
        speed_action = 0
    
    timings["choose_action"] = (time.time() - last_time) * 1000
    last_time = time.time()

    # --- 转换为游戏命令 ---
    # 方向映射
    direction_map = {
        0: "N",  # 不动
        1: "U",  # 上
        2: "D",  # 下
        3: "L",  # 左
        4: "R"   # 右
    }
    direction = direction_map.get(direction_action, "N")
    
    # 炸弹
    is_place_bomb = (bomb_action == 1)
    
    # 速度（相对速度档位转换为实际stride）
    from config import BASE_SPEED, SPEED_PER_BOOT, SPEED_GEAR_PERCENTAGES
    
    player = frame.my_player
    max_speed = BASE_SPEED + player.agility_boots_count * SPEED_PER_BOOT
    
    # 检查是否踩在加速点上
    player_pos = player.position
    corners = [
        (player_pos.x - 25, player_pos.y - 25), (player_pos.x + 24, player_pos.y - 25),
        (player_pos.x - 25, player_pos.y + 24), (player_pos.x + 24, player_pos.y + 24)
    ]
    
    on_acceleration = False
    for corner_x, corner_y in corners:
        grid_x = int(corner_x / 50)
        grid_y = int(corner_y / 50)
        if 0 <= grid_y < 16 and 0 <= grid_x < 28:
            terrain = frame.map[grid_y][grid_x].terrain
            if terrain == 'B':
                on_acceleration = True
                break
    
    if on_acceleration:
        max_speed *= 2.0
    
    # 根据速度档位计算实际stride
    if speed_action == 0:
        speed = 0  # stride=0表示最大速度
    else:
        percentage = SPEED_GEAR_PERCENTAGES.get(speed_action, 0.6)
        speed = int(max_speed * percentage)
        if speed == 0:
            speed = 1

    # --- 构造游戏响应 ---
    response_data = {
        "direction": direction,
        "is_place_bomb": is_place_bomb,
        "stride": speed
    }

    timings["translate_action"] = (time.time() - last_time) * 1000
    timings["total_time"] = (time.time() - start_time) * 1000
    
    # Debug输出：只打印超过阈值的慢响应
    if timings["total_time"] >= 85:
        print(f"⚠️ Tick {frame.current_tick} SLOW: { {k: f'{v:.2f}ms' for k, v in timings.items()} }")
    
    return jsonify(response_data)

@app.route("/api/v1/ping", methods=["HEAD"])
def handle_ping():
    return "", 200

def on_exit():
    """程序退出时的清理工作"""
    # 保存专家数据（如果有未保存的数据）
    if ENABLE_EXPERT_DATA_COLLECTION and expert_collector is not None:
        print(f"\n{'='*60}")
        print(f"程序退出：检查未保存的数据")
        print(f"{'='*60}")
        
        # 如果有未完成的episode，保存它
        if expert_collector.frame_count > 0:
            print(f"⚠️  发现未保存的数据 ({expert_collector.frame_count} 帧)")
            expert_collector.finish_episode()
        else:
            print(f"✅ 所有数据已保存")
        
        # 打印最终统计
        expert_collector.print_statistics()
        print(f"{'='*60}\n")

atexit.register(on_exit)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=False, threaded=False)
