import torch
import csv
import os
from datetime import datetime
from data_models import Frame
from agent import DQNAgent
from utils import preprocess_frame, MAP_WIDTH, MAP_HEIGHT
from model import DQN

class GameAI:
    def __init__(self):
        """Initializes the Game AI, including the DQN agent and state tracking."""
        # --- AI Agent Initialization ---
        STATE_CHANNELS = 11 # Updated to include invincibility status channel
        ACTION_SIZE = 6
        INPUT_SHAPE = (STATE_CHANNELS, MAP_HEIGHT, MAP_WIDTH)

        self.agent = DQNAgent(state_size=INPUT_SHAPE, action_size=ACTION_SIZE)
        try:
            self.agent.load("bomberman_dqn_2v2.pth")
            print("2v2 Model weights loaded successfully.")
        except FileNotFoundError:
            print("No pre-trained 2v2 model found, starting from scratch.")

        # --- State Tracking ---
        self.previous_state = None
        self.previous_action = None
        self.previous_frame_info = {
            'my_territory': 0,
            'enemy_territory': 0,
            'is_stunned': False,
            'items_collected': 0,
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

    def get_command(self, frame: Frame):
        """
        Processes a game frame and returns the AI's command.
        
        Args:
            frame (Frame): The current game frame data.
            
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
        
        action_probs_str = "RANDOM"
        if q_values is not None:
            # Apply softmax to convert Q-values to probabilities
            probs = torch.nn.functional.softmax(q_values, dim=1).squeeze().tolist()
            action_map = ["U", "D", "L", "R", "BOMB", "STAY"]
            action_probs_str = ", ".join([f"{action_map[i]}: {p:.2f}" for i, p in enumerate(probs)])

        print(f"--- Tick {frame.current_tick}: Action {action}, Reward {reward_str}, Total Reward: {self.total_reward:.2f}, Loss: {loss_str}, Epsilon {self.agent.epsilon:.4f} ---")
        print(f"    Probs: [ {action_probs_str} ]")

        # --- Append to Log File (Sampled every 10 ticks) ---
        if 'reward' in locals() and frame.current_tick % 10 == 0:
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([frame.current_tick, reward, self.total_reward, self.current_loss, self.agent.epsilon])

        return response_data

    def save_model(self):
        """Saves the agent's model weights."""
        print("--- Saving model weights on exit ---")
        self.agent.save("bomberman_dqn_2v2.pth")
        print("--- Model saved successfully ---")

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

        # Reward for collecting items
        current_items = frame.my_player.bomb_pack_count + frame.my_player.sweet_potion_count + frame.my_player.agility_boots_count
        if current_items > prev_info['items_collected']:
            reward += 50

        # Penalty for standing still
        if prev_info['frame'] is not None and frame.my_player.position == prev_info['frame'].my_player.position:
            reward -= 2 # Heavier penalty for not moving

        # Small penalty per tick to encourage action
        reward -= 1

        new_info = {
            'frame': frame,
            'my_territory': my_territory,
            'enemy_territory': enemy_territory,
            'is_stunned': is_currently_stunned,
            'items_collected': current_items
        }
        print(new_info['my_territory'], new_info['enemy_territory'])
        print(reward)
        return reward, new_info
