import torch
import numpy as np
import threading
from collections import deque
from config import FRAME_STACK_SIZE, ACTION_SPACE_DEFINITION

# --- Agent State Management ---

# This dictionary will store the state for each individual robot, identified by its robot_id.
# Using a dictionary allows the server to handle multiple robots simultaneously, each with its own data.
agent_states = {}

# A lock is crucial for thread safety. Since the web server might handle multiple requests
# concurrently in different threads, we need to prevent race conditions when accessing
# the shared `agent_states` dictionary.
agent_states_lock = threading.Lock()

def get_or_create_agent_state(robot_id):
    """
    Retrieves or initializes the state for a given robot_id in a thread-safe manner.

    If the robot_id is new, it creates a default state structure for it, which includes:
    - "raw_frame_buffer": Stores raw Frame objects, mainly for reward calculation.
    - "processed_frame_buffer": Stores preprocessed observations for model input.
    - "trajectory_buffer": A list to accumulate experiences (state, action, reward, done)
      before sending them for training.
    - "last_action": Stores the action taken in the previous step, which is needed to form
      a complete experience tuple.

    Args:
        robot_id (str): The unique identifier for the robot.

    Returns:
        dict: The state dictionary for the specified robot.
    """
    with agent_states_lock:
        if robot_id not in agent_states:
            agent_states[robot_id] = {
                "raw_frame_buffer": deque(maxlen=FRAME_STACK_SIZE),
                "processed_frame_buffer": deque(maxlen=FRAME_STACK_SIZE),
                "trajectory_buffer": [],
                "last_action": np.zeros(len(ACTION_SPACE_DEFINITION), dtype=int),
                "last_value": torch.tensor([0.0]),
                "last_log_prob": torch.tensor([0.0]) # Initialize with a valid tensor
            }
        return agent_states[robot_id]

def clear_agent_state(robot_id):
    """
    Clears the state buffers for a specific agent, typically after an episode ends.
    This is important to prevent data from one episode from leaking into the next.

    Args:
        robot_id (str): The unique identifier for the robot.
    """
    with agent_states_lock:
        if robot_id in agent_states:
            agent_states[robot_id]["raw_frame_buffer"].clear()
            agent_states[robot_id]["processed_frame_buffer"].clear()
            agent_states[robot_id]["trajectory_buffer"].clear()
            # last_action can be reset or left as is, depending on strategy
