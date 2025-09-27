import torch
import numpy as np
from core_model import preprocess_observation_dict
from reward import calculate_reward
from agent_manager import clear_agent_state
from config import FRAME_STACK_SIZE, TRAIN_BATCH_SIZE

# --- MODIFIABLE STRATEGY SECTION ---
# This is the primary file for you to modify and experiment with your strategy.

def stack_observations(processed_frame_buffer):
    """
    Stacks already preprocessed observation dictionaries from the buffer.
    This function handles the multi-input architecture by stacking each component separately.
    """
    # Stack the grid views
    stacked_grid_view = np.concatenate([obs['grid_view'] for obs in processed_frame_buffer], axis=0)
    
    # Stack the pixel views
    stacked_pixel_view = np.concatenate([obs['pixel_view'] for obs in processed_frame_buffer], axis=0)

    # For player state, we only use the most recent frame's data
    latest_player_state = processed_frame_buffer[-1]['player_state']
    
    return {
        "grid_view": stacked_grid_view,
        "pixel_view": stacked_pixel_view,
        "player_state": latest_player_state
    }

def decide_action(state, model):
    """
    Decides an action and records the associated value and log_prob for PPO training.
    """
    processed_frame_buffer = state['processed_frame_buffer']
    
    if len(processed_frame_buffer) < FRAME_STACK_SIZE:
        # Return default action (Stop, No Bomb) and placeholder tensors
        return np.array([0, 0]), torch.tensor([0.0]), torch.tensor([0.0])
    
    stacked_observation = stack_observations(processed_frame_buffer)
    
    # Get action, value, and log_prob from the model
    # We need to convert the observation to tensors and add a batch dimension
    with torch.no_grad():
        obs_tensor = {k: torch.as_tensor(v).unsqueeze(0).to(model.device) for k, v in stacked_observation.items()}
        action, value, log_prob = model.policy.forward(obs_tensor)

    action = action.numpy()[0]
    
    # Move value and log_prob to CPU before returning, so they can be stored in numpy-based buffers
    return action, value, log_prob

def collect_experience(robot_id, state, current_frame, trajectory_queue):
    """
    Calculates reward and collects the experience tuple for training.

    YOU CAN MODIFY THIS:
    - The `calculate_reward` function is imported from `reward.py`. You can
      implement your own reward logic there.
    - You could add logic here to modify the reward based on the agent's
      trajectory buffer or other long-term metrics.
    """
    # Experience collection only happens when we have enough frames to form a
    # complete "before" and "after" picture.
    if len(state['processed_frame_buffer']) == FRAME_STACK_SIZE:
        # The "previous state" is constructed from the processed buffer.
        # It's the first (N-1) frames plus the second-to-last frame, representing
        # the state right before the last action was taken.
        last_stacked_obs = stack_observations(list(state['processed_frame_buffer'])[:-1] + [state['processed_frame_buffer'][-2]])
        
        # --- REWARD CALCULATION ---
        # This is a key part of your strategy. Modify `reward.py` to change how
        # the agent is incentivized. It uses the raw frame buffer for accurate state comparison.
        reward = calculate_reward(current_frame, state['raw_frame_buffer'][-2])
        
        # An episode is "done" if the player is dead OR the game timer runs out.
        # Combining both conditions makes the logic more robust.
        done = current_frame.my_player.status == 'dead' or (current_frame.current_tick >= 1800)
        
        # The complete experience tuple for PPO, including the "decision memory"
        experience = (
            last_stacked_obs,
            state['last_action'],
            reward,
            done,
            {}, # info dict
            state['last_value'],
            state['last_log_prob']
        )
        state['trajectory_buffer'].append(experience)
        
        # If the trajectory buffer is full or the episode is over, submit it for training.
        if done or len(state['trajectory_buffer']) >= TRAIN_BATCH_SIZE:
            # The trajectory buffer is now full or the episode is over.
            # We submit a copy of the buffer to the central training queue.
            # The content being submitted is a list of experience tuples, where each tuple is:
            # (observation, action, reward, done, info_dict)
            # - observation: A dictionary {"map": stacked_map, "player_state": player_state}
            # - action: The integer ID of the action taken.
            # - reward: The float reward calculated for this step.
            # - done: A boolean indicating if the episode ended.
            # - info_dict: An empty dictionary (can be used for debugging).
            print(f"Agent {robot_id}'s trajectory is full or episode ended. Submitting to training queue.")
            trajectory_queue.put(list(state['trajectory_buffer']))  # Submit a copy
            state['trajectory_buffer'].clear()

        if done:
            # If the episode is over, clear the agent's state to start fresh.
            clear_agent_state(robot_id)

def action_to_response(action_id):
    """
    Converts a numeric action ID from the model into a game-understandable command.
    """
    # This mapping assumes a MultiDiscrete([5, 2]) action space
    # Part 1: Movement (0: Stop, 1: Up, 2: Down, 3: Left, 4: Right)
    # Part 2: Bomb (0: No, 1: Yes)
    
    move_action = action_id[0]
    bomb_action = action_id[1]

    direction_map = {0: "N", 1: "U", 2: "D", 3: "L", 4: "R"}
    
    direction = direction_map.get(move_action, "N")
    is_place_bomb = bool(bomb_action == 1)
    
    # Stride is 1 if moving, 0 if standing still
    stride = 0
    
    return {"direction": direction, "is_place_bomb": is_place_bomb, "stride": stride}
