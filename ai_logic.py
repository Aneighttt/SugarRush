import torch
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
            'items_collected': 0
        }

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
            done = False  # Game doesn't end on stun
            
            self.agent.remember(self.previous_state, self.previous_action, reward, current_state_tensor, done)
            self.previous_frame_info = new_frame_info
            
            self.agent.replay(batch_size=32)
        else:
            # Initialize frame info on the first frame
            my_territory, enemy_territory = self._count_territory(frame)
            self.previous_frame_info['my_territory'] = my_territory
            self.previous_frame_info['enemy_territory'] = enemy_territory

        action = self.agent.choose_action(current_state_tensor)

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
        print(f"--- Tick {frame.current_tick}: Action {action}, Reward {reward_str}, Epsilon {self.agent.epsilon:.4f} ---")

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
        for row in frame.map:
            for cell in row:
                if cell.ownership == my_team_id:
                    my_territory += 1
                elif cell.ownership is not None:
                    enemy_territory += 1
        return my_territory, enemy_territory

    def _calculate_reward(self, frame: Frame, prev_info: dict):
        reward = 0
        my_territory, enemy_territory = self._count_territory(frame)

        prev_diff = prev_info['my_territory'] - prev_info['enemy_territory']
        current_diff = my_territory - enemy_territory
        reward += (current_diff - prev_diff) * 5

        is_currently_stunned = 'INVINCIBLE' in [s.name for s in frame.my_player.extra_status]
        if is_currently_stunned and not prev_info['is_stunned']:
            reward -= 300

        current_items = frame.my_player.bomb_pack_count + frame.my_player.sweet_potion_count + frame.my_player.agility_boots_count
        if current_items > prev_info['items_collected']:
            reward += 50

        reward += 0.1

        new_info = {
            'my_territory': my_territory,
            'enemy_territory': enemy_territory,
            'is_stunned': is_currently_stunned,
            'items_collected': current_items
        }
        return reward, new_info
