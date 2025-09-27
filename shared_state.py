import os
import queue
import numpy as np
import torch
from stable_baselines3 import PPO
import gymnasium as gym
from config import *

# --- MODIFIABLE PARAMETERS ---
# This section contains parameters you are likely to tune for your strategy.



# --- SHARED RESOURCES (Advanced) ---
# These are core components of the RL system. Modify with caution.
# This queue will be created in the main entry point and passed around.

# --- Observation and Action Spaces ---
# This defines the structure of the input the model will receive.
# We are using a multi-input architecture with two main components:
# 1. "grid_view": A low-resolution tactical grid (11x11xChannels).
# 2. "pixel_view": A high-resolution pixel grid for precise micro-control (100x100xChannels).
# 3. "player_state": A vector of numerical features.
PIXEL_VIEW_SIZE = 100
PIXEL_CHANNELS = 2 # 0: Terrain, 1: Danger Zone
STACKED_PIXEL_CHANNELS = PIXEL_CHANNELS * FRAME_STACK_SIZE

observation_space = gym.spaces.Dict({
    # Low-resolution tactical grid
    "grid_view": gym.spaces.Box(
        low=0, high=2, shape=(STACKED_MAP_CHANNELS, MAP_HEIGHT, MAP_WIDTH), dtype=np.float32
    ),
    # High-resolution pixel grid for micro-control
    "pixel_view": gym.spaces.Box(
        low=0, high=1.0, shape=(STACKED_PIXEL_CHANNELS, PIXEL_VIEW_SIZE, PIXEL_VIEW_SIZE), dtype=np.float32
    ),
    # Vector of player stats
    "player_state": gym.spaces.Box(
        low=0, high=np.inf, shape=(PLAYER_STATE_SIZE,), dtype=np.float32
    )
})

# Defines the set of possible actions the agent can take.
action_space = gym.spaces.MultiDiscrete(ACTION_SPACE_DEFINITION)


# --- Dummy Environment for Model Initialization ---
# This is a workaround for some versions of Stable Baselines3 that require a valid
# environment during initialization, instead of accepting `env=None` with separate space arguments.
class DummyEnv(gym.Env):
    """A dummy environment that has the observation and action spaces we want."""
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

    def step(self, action):
        # This method is not used for the disembodied model, but is required by the interface.
        obs = self.observation_space.sample()
        return obs, 0, False, False, {}

    def reset(self, seed=None, options=None):
        # This method is not used, but is required by the interface.
        return self.observation_space.sample(), {}


# --- SHARED PPO MODEL ---
# This is the central "brain". It's loaded once and used by all agents for decisions.
# The training worker updates this model in the background.
def load_or_create_model():
    """
    Loads a pre-existing PPO model or creates a new one, automatically selecting the best device.
    """

    # Create a dummy environment with the correct spaces. This is needed for both creating and loading.
    dummy_env = DummyEnv(observation_space, action_space)

    print("Creating a new model")
    model = PPO(
        "MultiInputPolicy",
        dummy_env,
        n_steps=512,
        batch_size = 128,
        learning_rate =3e-4,
        device='cpu',
        verbose = 1,
        tensorboard_log = "./log/"
    )
    return model

# The global model instance is no longer created here.
# It will be created in the main application entry point (robot.py)
# and passed to the functions that need it.
