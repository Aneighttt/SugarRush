import torch
import numpy as np
import csv
import os
from datetime import datetime
from data_models import Frame
from agent import DQNAgent
from utils import preprocess_frame, MAP_WIDTH, MAP_HEIGHT
from model import DQN

class GameAI:
    def __init__(self, agent: DQNAgent):
        """
        Initializes the Game AI.
        This class now acts as a player instance that uses a shared, persistent agent.
        
        Args:
            agent (DQNAgent): The shared DQN agent that holds the model and learning state.
        """
        # --- AI Agent Reference ---
        self.agent = agent

        # --- State Tracking (specific to this player instance for one game) ---
        self.previous_state = None
        self.previous_action = None
        self.previous_frame_info = {
            'my_territory': 0,
            'enemy_territory': 0,
            'is_stunned': False,
            'items_collected': 0,
            'my_bomb_identifiers': set(),
            'frame': None
        }
        self.total_reward = 0
        self.current_loss = 0

        # --- Logging Initialization ---
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"training_log_{timestamp}.csv")

        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Tick", "Reward", "TotalReward", "Loss", "Epsilon"])

    def get_command(self, frame: Frame, player_id: str):
        """
        Processes a game frame and returns the AI's command.
        
        Args:
            frame (Frame): The current game frame data.
            player_id (str): The ID of the player for logging purposes.
            
        Returns:
            dict: A dictionary containing the command for the game server.
        """
        current_state = preprocess_frame(frame)
        current_state_tensor = torch.from_numpy(current_state).float().unsqueeze(0)

        if self.previous_state is not None:
            reward, new_frame_info = self._calculate_reward(frame, self.previous_frame_info)
            
            # Check for the end of the game
            done = frame.current_tick == 1800
            
            self.agent.remember(self.previous_state, self.previous_action, reward, current_state_tensor, done)
            self.previous_frame_info = new_frame_info
            
            loss = self.agent.replay(batch_size=32)
            if loss is not None:
                self.current_loss = loss
            self.total_reward += reward
        else:
            # Initialize frame info on the first frame
            my_territory, enemy_territory = self._count_territory(frame)
            self.previous_frame_info['my_territory'] = my_territory
            self.previous_frame_info['enemy_territory'] = enemy_territory

        action, q_values = self.agent.choose_action(current_state_tensor)

        # Translate action to game command
        direction = "N"
        is_place_bomb = False
        if action == 0: direction = "U"
        elif action == 1: direction = "D"
        elif action == 2: direction = "L"
        elif action == 3: direction = "R"
        elif action == 4:
            my_bombs_on_map = sum(1 for bomb in frame.bombs if bomb.owner_id == frame.my_player.id)
            if my_bombs_on_map < frame.my_player.bomb_pack_count:
                is_place_bomb = True
        
        speed = 10 + frame.my_player.agility_boots_count * 2
        
        response_data = {
            "direction": direction,
            "is_place_bomb": is_place_bomb,
            "stride": speed
        }
        
        self.previous_state = current_state_tensor
        self.previous_action = action
        
        # For logging purposes
        reward_str = f"{reward:.2f}" if 'reward' in locals() else "N/A"
        loss_str = f"{self.current_loss:.4f}" if self.current_loss != 0 else "N/A"
        
        action_map = ["U", "D", "L", "R", "BOMB", "STAY"]
        action_probs_str = "RANDOM"
        is_random_action = q_values is None
        
        if not is_random_action:
            # Apply softmax to convert Q-values to probabilities
            probs = torch.nn.functional.softmax(q_values, dim=1).squeeze().tolist()
            action_probs_str = ", ".join([f"{action_map[i]}: {p:.2f}" for i, p in enumerate(probs)])

        print(f"--- [Player {player_id}] Tick {frame.current_tick}: Action {action}, Reward {reward_str}, Total Reward: {self.total_reward:.2f}, Loss: {loss_str}, Epsilon {self.agent.epsilon:.4f} ---")
        print(f"    [Player {player_id}] Probs: [ {action_probs_str} ]")

        # --- Append to Log File (Sampled every 10 ticks) ---
        if 'reward' in locals() and frame.current_tick % 10 == 0:
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([frame.current_tick, reward, self.total_reward, self.current_loss, self.agent.epsilon])

        # --- Prepare Visualization Data ---
        action_name = action_map[action]
        if is_random_action:
            action_name += " (Random)"

        viz_data = {
            "player_id": player_id,
            "tick": frame.current_tick,
            "action": action_name,
            "reward": float(f"{reward:.2f}") if 'reward' in locals() else None,
            "total_reward": float(f"{self.total_reward:.2f}"),
            "loss": float(f"{self.current_loss:.4f}") if self.current_loss != 0 else None,
            "epsilon": float(f"{self.agent.epsilon:.4f}"),
            "q_values": [float(f"{p:.2f}") for p in probs] if not is_random_action else None
        }

        return response_data, viz_data

    # The save_model method is no longer needed here, as the shared agent
    # will be saved directly by the main robot script.

    def _count_territory(self, frame: Frame):
        my_territory = 0
        enemy_territory = 0
        my_team_id = frame.my_player.team
        
        # Find the enemy team ID by looking for the first player who is not on my team
        enemy_team_id = next((player.team for player in frame.other_players if player.team != my_team_id), 'N')
        for row in frame.map:
            for cell in row:
                if cell.ownership == my_team_id:
                    my_territory += 1
                elif cell.ownership == enemy_team_id:
                    enemy_territory += 1
        return my_territory, enemy_territory

    def get_occupied_grids_from_position(self, pixel_pos):
        """
        Calculates all grid cells that a 50x50 player body overlaps with.
        This is a helper function for reward calculation.
        """
        PIXEL_PER_CELL = 50
        center_x, center_y = pixel_pos.x, pixel_pos.y
        min_x, max_x = center_x - 25, center_x + 24
        min_y, max_y = center_y - 25, center_y + 24

        start_gx = int(min_x / PIXEL_PER_CELL)
        end_gx = int(max_x / PIXEL_PER_CELL)
        start_gy = int(min_y / PIXEL_PER_CELL)
        end_gy = int(max_y / PIXEL_PER_CELL)

        occupied = set()
        for gx in range(start_gx, end_gx + 1):
            for gy in range(start_gy, end_gy + 1):
                if 0 <= gx < MAP_WIDTH and 0 <= gy < MAP_HEIGHT:
                    occupied.add((gx, gy))
        return list(occupied)

    def _get_explosion_grids(self, bomb, frame: Frame):
        """
        Calculates the grid cells affected by a single bomb's explosion,
        correctly handling all types of obstacles.
        """
        explosion_grids = set()
        bomb_x, bomb_y = bomb.position.x, bomb.position.y
        
        explosion_grids.add((bomb_x, bomb_y))

        # --- Check in 4 directions using server's grid coordinates ---
        # Right
        for i in range(1, bomb.range + 1):
            x = bomb_x + i
            if not (0 <= x < MAP_WIDTH): break
            explosion_grids.add((x, bomb_y))
            if frame.map[bomb_y][x].terrain in ['I', 'N', 'D']: break
        # Left
        for i in range(1, bomb.range + 1):
            x = bomb_x - i
            if not (0 <= x < MAP_WIDTH): break
            explosion_grids.add((x, bomb_y))
            if frame.map[bomb_y][x].terrain in ['I', 'N', 'D']: break
        # Up
        for i in range(1, bomb.range + 1):
            y = bomb_y + i
            if not (0 <= y < MAP_HEIGHT): break
            explosion_grids.add((bomb_x, y))
            if frame.map[y][bomb_x].terrain in ['I', 'N', 'D']: break
        # Down
        for i in range(1, bomb.range + 1):
            y = bomb_y - i
            if not (0 <= y < MAP_HEIGHT): break
            explosion_grids.add((bomb_x, y))
            if frame.map[y][bomb_x].terrain in ['I', 'N', 'D']: break
            
        return explosion_grids

    def get_danger_zones(self, frame: Frame):
        """
        Calculates all grid cells that are currently within any bomb's explosion range.
        """
        danger_zones = set()
        for bomb in frame.bombs:
            danger_zones.update(self._get_explosion_grids(bomb, frame))
        return danger_zones

    def _calculate_reward(self, frame: Frame, prev_info: dict):
        reward = 0
        my_territory, enemy_territory = self._count_territory(frame)

        # --- Terminal Reward on the last tick ---
        if frame.current_tick == 1800:
            if my_territory > enemy_territory:
                reward += 500  # Large reward for winning
            elif my_territory < enemy_territory:
                reward -= 500  # Large penalty for losing
            # No extra reward for a draw

        # Reward for territory change
        prev_diff = prev_info['my_territory'] - prev_info['enemy_territory']
        current_diff = my_territory - enemy_territory
        reward += (current_diff - prev_diff) * 5
        
        # Penalty for getting stunned is now implicitly handled by the large territory loss that follows.
        is_currently_stunned = (frame.my_player.status == 'D')
        if is_currently_stunned and not prev_info['is_stunned']: # Penalize only on the frame it happens
            reward -= 200 # Heavy penalty for getting stunned/killed

        # Reward for collecting items
        current_items = frame.my_player.bomb_pack_count + frame.my_player.sweet_potion_count + frame.my_player.agility_boots_count
        if current_items > prev_info['items_collected']:
            reward += 50

        # --- Reward for strategic bomb placement (Intent-based reward) ---
        if self.previous_action == 4 and prev_info['frame'] is not None:
            my_current_bombs = [b for b in frame.bombs if b.owner_id == frame.my_player.id]
            current_bomb_identifiers = {(b.position.x, b.position.y, b.explode_at) for b in my_current_bombs}
            prev_bomb_identifiers = prev_info.get('my_bomb_identifiers', set())
            
            new_bomb_identifiers = current_bomb_identifiers - prev_bomb_identifiers

            if new_bomb_identifiers:
                # Find the actual bomb object corresponding to the new identifier
                new_bomb_identifier = new_bomb_identifiers.pop()
                new_bomb = next((b for b in my_current_bombs if (b.position.x, b.position.y, b.explode_at) == new_bomb_identifier), None)

                if new_bomb:
                    strategic_value = 0
                    explosion_area = self._get_explosion_grids(new_bomb, frame)
                    
                    enemy_grids = set()
                    my_team_id = frame.my_player.team
                    for other_player in frame.other_players:
                        if other_player.team != my_team_id:
                            for gx, gy in self.get_occupied_grids_from_position(other_player.position):
                                enemy_grids.add((gx, gy))

                    for x, y in explosion_area:
                        if frame.map[y][x].terrain == 'D':
                            strategic_value += 2
                        if (x, y) in enemy_grids:
                            strategic_value += 15
                    
                    if strategic_value > 0:
                        reward += strategic_value

        # --- Penalties ---

        # 1. Penalty for ineffective movement (bumping into walls)
        # This happens if a move action was issued but the agent's position did not change.
        is_move_action = self.previous_action in [0, 1, 2, 3]
        if prev_info['frame'] is not None and is_move_action and frame.my_player.position == prev_info['frame'].my_player.position:
            reward -= 10  # Heavy penalty for bumping into a wall

        # 2. Penalty for intentionally standing still
        # This penalty must be high to force the agent to explore.
        # It should be comparable to the penalty for bumping into a wall.
        if self.previous_action == 5: # Assuming 5 is the "STAY" action
            reward -= 10 # Heavy penalty to discourage idling

        # 3. Small penalty per tick to encourage action and speed up the game.
        reward -= 0.1

        # 4. Penalty for standing in a bomb's danger zone
        # We need to check the player's occupied grids against the danger zones
        player_grids = self.get_occupied_grids_from_position(frame.my_player.position)
        danger_zones = self.get_danger_zones(frame)
        is_in_danger = any(grid in danger_zones for grid in player_grids)
        if is_in_danger:
            reward -= 5 # Penalty for staying in a dangerous area

        new_info = {
            'frame': frame,
            'my_territory': my_territory,
            'enemy_territory': enemy_territory,
            'is_stunned': is_currently_stunned,
            'items_collected': current_items,
            'my_bomb_identifiers': {(b.position.x, b.position.y, b.explode_at) for b in frame.bombs if b.owner_id == frame.my_player.id}
        }
        return reward, new_info
