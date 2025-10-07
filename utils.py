import collections
import numpy as np
from config import *
from data_models import Frame

def calculate_distance_map_to_frontier(frame: Frame):
    """
    Calculates a map of shortest path distances from every walkable tile
    to the nearest non-friendly tile using a single, efficient BFS starting
    from all targets simultaneously.
    """
    my_team_id = frame.my_player.team
    
    q = collections.deque()
    dist_map = np.full((MAP_HEIGHT, MAP_WIDTH), -1, dtype=int)

    # 1. Initialize the queue with all target tiles (the "frontier") at distance 0.
    for y in range(MAP_HEIGHT):
        for x in range(MAP_WIDTH):
            cell = frame.map[y][x]
            if cell.ownership != my_team_id and cell.terrain not in ['I', 'N', 'D']:
                q.append((x, y, 0))
                dist_map[y, x] = 0

    # 2. Run a single BFS. The first time a tile is visited, it's guaranteed
    #    to be via the shortest path from one of the nearest targets.
    while q:
        x, y, dist = q.popleft()
        
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            
            # If the neighbor is within bounds and has not been visited yet...
            if 0 <= nx < MAP_WIDTH and 0 <= ny < MAP_HEIGHT and dist_map[ny, nx] == -1:
                # ...and is walkable...
                if frame.map[ny][nx].terrain not in ['I', 'N', 'D']:
                    dist_map[ny, nx] = dist + 1
                    q.append((nx, ny, dist + 1))
    
    return dist_map

def get_color_escape(r, g, b, background=False):
    """Returns the ANSI escape code for a given RGB color."""
    return f'\033[{"48" if background else "38"};2;{r};{g};{b}m'

def visualize_gradient_in_terminal(view_gradient_map, frame: Frame):
    """
    Renders a high-contrast 11x11 view of GRADIENT values for dark terminals.
    """
    # --- ANSI Color Definitions for High Contrast (Dark Terminal Friendly) ---
    RESET = '\033[0m'
    WHITE_TEXT = get_color_escape(255, 255, 255)
    
    PLAYER_BG = get_color_escape(138, 43, 226, background=True)  # Bright Purple
    WALL_BG = get_color_escape(80, 80, 80, background=True)      # Dark Grey
    TARGET_BG = get_color_escape(0, 255, 255, background=True)    # Bright Cyan

    # --- Header ---
    header_width = VIEW_SIZE * 7
    print("\n" + "="*header_width)
    print(f"--- 11x11 View Gradient Map (Tick: {frame.current_tick}) ---".center(header_width))
    print(f"{PLAYER_BG}{WHITE_TEXT}  PP   {RESET} Player | {WALL_BG}{WHITE_TEXT}  ###  {RESET} Wall | {TARGET_BG}{WHITE_TEXT} 1.00  {RESET} Target".center(header_width))
    print("-" * header_width)

    # --- Map Rendering ---
    player_grid_x, player_grid_y = frame.my_player.position.x // PIXEL_PER_CELL, frame.my_player.position.y // PIXEL_PER_CELL
    view_center_x, view_center_y = VIEW_SIZE // 2, VIEW_SIZE // 2

    # Invert the y-axis during printing to match the game's coordinate system (0,0 at bottom-left).
    for view_y in range(VIEW_SIZE - 1, -1, -1):
        line = ""
        for view_x in range(VIEW_SIZE):
            if (view_x, view_y) == (view_center_x, view_center_y):
                line += f"{PLAYER_BG}{WHITE_TEXT}  PP   {RESET}"
                continue

            map_x = player_grid_x + (view_x - view_center_x)
            map_y = player_grid_y + (view_y - view_center_y)

            if not (0 <= map_y < MAP_HEIGHT and 0 <= map_x < MAP_WIDTH):
                line += f"{WALL_BG}{WHITE_TEXT}  ###  {RESET}"
                continue

            cell = frame.map[map_y][map_x]
            if cell.terrain in ['I', 'N', 'D']:
                line += f"{WALL_BG}{WHITE_TEXT}  ###  {RESET}"
            else:
                gradient = view_gradient_map[view_y, view_x]
                if gradient == 1.0:
                    line += f"{TARGET_BG}{WHITE_TEXT} 1.00  {RESET}"
                elif gradient > 0:
                    # Use text color to represent gradient strength
                    if gradient > 0.66:
                        color = get_color_escape(0, 255, 0) # Bright Green
                    elif gradient > 0.33:
                        color = get_color_escape(255, 255, 0) # Bright Yellow
                    else:
                        color = get_color_escape(255, 0, 0) # Bright Red
                    line += f"{color}{gradient:^7.2f}{RESET}"
                else:
                    line += "   .   " # No gradient
        print(line)
    print("="*header_width + "\n")
