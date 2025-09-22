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
# The input is now a self-centered 11x11 view, stacked over 2 frames,
# plus a vector of non-visual information.
VIEW_SIZE = 11
FRAME_STACK = 2
VISUAL_STATE_CHANNELS = 11 * FRAME_STACK
VECTOR_STATE_SIZE = 3 # agility_boots, bomb_pack, sweet_potion
ACTION_SIZE = 6
INPUT_SHAPE = (VISUAL_STATE_CHANNELS, VIEW_SIZE, VIEW_SIZE)
shared_agent = DQNAgent(state_size=INPUT_SHAPE, action_size=ACTION_SIZE, vector_size=VECTOR_STATE_SIZE)
try:
    shared_agent.load("bomberman_dqn_2v2.pth")
    print("--- Shared agent model weights loaded successfully. ---")
except FileNotFoundError:
    print("--- No pre-trained model found for shared agent, starting from scratch. ---")


# --- Per-Game Player Instance Management ---
game_players = {}
current_game_tick = -1
tactical_info_buffer = {} # Buffer to hold tactical data for the current tick

# --- Terminal Visualization ---
from termcolor import colored
from utils import get_grid_position

def render_tactical_sandboard(frame: Frame, tactical_data: dict):
    """Renders a beautiful and intuitive tactical sandboard to the terminal."""
    # --- Color Palette (Optimized for light terminals) ---
    COLORS = {
        'I': 'on_dark_grey', 'N': 'on_dark_grey',
        'D': 'on_light_yellow',
        'P': 'on_white',
        'B': 'on_cyan',
        'M': 'on_magenta',
        'BOMB': 'on_red',
        'ITEM': 'on_green',
    }
    
    grid_chars = [['  ' for _ in range(MAP_WIDTH)] for _ in range(MAP_HEIGHT)]
    grid_bg = [['on_white' for _ in range(MAP_WIDTH)] for _ in range(MAP_HEIGHT)]
    grid_fg = [['black' for _ in range(MAP_WIDTH)] for _ in range(MAP_HEIGHT)]

    # 1. Draw Terrain
    for y, row in enumerate(frame.map):
        for x, cell in enumerate(row):
            if cell.terrain in COLORS:
                grid_bg[y][x] = COLORS[cell.terrain]

    # 2. Draw Items & Bombs
    for item in frame.map_items:
        grid_bg[item.position.y][item.position.x] = COLORS['ITEM']
        # grid_chars[item.position.y][item.position.x] = '()'
    for bomb in frame.bombs:
        grid_bg[bomb.position.y][bomb.position.x] = COLORS['BOMB']
        # grid_chars[bomb.position.y][bomb.position.x] = '@@'

    # 3. Draw Players
    all_players = [frame.my_player] + frame.other_players
    player_colors = {}
    my_team_id = frame.my_player.team
    
    for p in all_players:
        color = 'blue' if p.team == my_team_id else 'red'
        player_colors[p.id] = color
        gx, gy = get_grid_position(p.position)
        if 0 <= gx < MAP_WIDTH and 0 <= gy < MAP_HEIGHT:
            grid_chars[gy][gx] = f'P{p.id}'
            grid_fg[gy][gx] = color

    # 4. Draw Q-Values (as two digits to prevent deformation)
    for player_id, data in tactical_data.items():
        if data and data['q_values']:
            q = data['q_values']
            gx, gy = get_grid_position(data['position'])
            color = player_colors.get(player_id, 'white')
            
            if gy + 1 < MAP_HEIGHT and grid_chars[gy+1][gx] == '  ': grid_chars[gy+1][gx] = f"{int(q[0]*100):02d}"
            if gy - 1 >= 0 and grid_chars[gy-1][gx] == '  ': grid_chars[gy-1][gx] = f"{int(q[1]*100):02d}"
            if gx - 1 >= 0 and grid_chars[gy][gx-1] == '  ': grid_chars[gy][gx-1] = f"{int(q[2]*100):02d}"
            if gx + 1 < MAP_WIDTH and grid_chars[gy][gx+1] == '  ': grid_chars[gy][gx+1] = f"{int(q[3]*100):02d}"

    # 5. Assemble Final String
    os.system('cls' if os.name == 'nt' else 'clear')
    output = f"--- Tick: {frame.current_tick} ---\n"
    for y in range(MAP_HEIGHT - 1, -1, -1):
        for x in range(MAP_WIDTH):
            output += colored(grid_chars[y][x], grid_fg[y][x], grid_bg[y][x])
        output += "\n"
    
    # 6. Display Detailed Q-Values Below the Map
    details_str = ""
    action_map = ["U", "D", "L", "R", "BOMB", "STAY"]
    my_team_player_ids = {p.id for p in [frame.my_player] + frame.other_players if p.team == frame.my_player.team}

    for player_id in sorted(list(my_team_player_ids)):
        data = tactical_data.get(player_id)
        color = player_colors.get(player_id, 'white')
        details_str += colored(f"--- Player {player_id} (Elapsed: {data['elapsed_ms']:.2f}ms) --- | ", color, attrs=['bold']) if data else ""
        if data and data['q_values']:
            q = data['q_values']
            for i, action_name in enumerate(action_map):
                details_str += colored(f"{action_name}: {q[i]:.2f} | ", color)
        elif data:
            details_str += colored("RANDOM ACTION", color)
        details_str += "\n"

    output += details_str
    print(output)


@app.route("/api/v1/command", methods=["POST"])
def handle_command():
    """
    Handles the command request from the game server.
    It detects new games, manages temporary player instances,
    and returns the command from the shared AI brain.
    """
    global current_game_tick, game_players, shared_agent, tactical_info_buffer
    start_time = time.time()

    data = request.get_json()
    frame = Frame(data)
    player_id = frame.my_player.id

    # --- Game Reset Detection ---
    if frame.current_tick < 10 and current_game_tick > 1790:
        shared_agent.save("bomberman_dqn_2v2.pth")
        game_players.clear()
        tactical_info_buffer.clear()
    
    current_game_tick = max(current_game_tick, frame.current_tick)

    # Get or create a temporary player instance
    if player_id not in game_players:
        game_players[player_id] = GameAI(agent=shared_agent)
    
    player_instance = game_players[player_id]
    
    # Get command and tactical data from the AI
    response_data, viz_data, tactical_data = player_instance.get_command(frame, player_id)
    
    end_time = time.time()
    elapsed_ms = (end_time - start_time) * 1000
    
    # Inject elapsed_ms into the tactical data
    if tactical_data:
        tactical_data['elapsed_ms'] = elapsed_ms

    # Store tactical data in the buffer
    tactical_info_buffer[player_id] = tactical_data
    
    # --- Render Sandboard if we have info from all our bots ---
    # This assumes we know the number of bots we are controlling.
    # For a 2v2, if we control 2 bots, we wait for 2 data points.
    my_team_player_ids = {p.id for p in [frame.my_player] + frame.other_players if p.team == frame.my_player.team}
    if all(pid in tactical_info_buffer for pid in my_team_player_ids):
        render_tactical_sandboard(frame, tactical_info_buffer)
        tactical_info_buffer.clear() # Clear buffer for the next tick

    # Write visualization data to a file (optional)
    # log_dir = "logs"
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir)
    # with open(os.path.join(log_dir, f"viz_{player_id}.json"), "w") as f:
    #     json.dump(viz_data, f)
    
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
