import torch
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.buffers import DictReplayBuffer
import os

from environment import BomberEnv


class RealActionReplayBuffer(DictReplayBuffer):
    def add(self, obs, next_obs, action, reward, done, infos):
        # When using vectorized environments, infos is a list of dicts.
        # We need to extract the "real_action" from each info dict.
        real_actions = np.array([info["real_action"] for info in infos])
        super().add(obs, next_obs, real_actions, reward, done, infos)

class SB3_DQNAgent:
    def __init__(self, env: BomberEnv, log_dir="./log/"):
        """
        Initializes a DQNAgent that wraps a Stable Baselines 3 DQN model.
        The agent is now tightly coupled with a Gym environment.

        Args:
            env (BomberEnv): The custom Gym environment.
            log_dir (str): Directory to save tensorboard logs.
        """
        # if torch.backends.mps.is_available():
        #     device = torch.device("mps")
        # elif torch.cuda.is_available():
        #     device = torch.device("cuda")
        # else:
        #     device = torch.device("cpu")
        device = torch.device("cpu")
        self.model = DQN(
            policy="MultiInputPolicy",
            env=env,
            learning_rate=lambda p: 0.0001 + (0.001 - 0.0001) * p,  # Linear decay from 0.001 to 0.0001
            buffer_size=20000,
            learning_starts=2000, # Start training after 100 steps
            replay_buffer_class=RealActionReplayBuffer,
            batch_size=64,
            gamma=0.95,
            train_freq=(1, "step"),
            gradient_steps=1,
            target_update_interval=1800,
            exploration_fraction=0.1,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.1,
            max_grad_norm=10,  # Add gradient clipping to prevent explosion
            verbose=1,
            tensorboard_log=log_dir,
            device=device
        )

    def choose_action(self, observation, deterministic=False):
        """
        Chooses an action using the model's prediction.

        Args:
            observation: The current observation from the environment.
            deterministic (bool): Whether to use a deterministic policy.
        """
        action, _states = self.model.predict(observation, deterministic=deterministic)
        return action.item()

    def load(self, path, load_replay_buffer=True, fine_tuning=False):
        """
        Loads a pre-trained model and optionally its replay buffer.

        Args:
            path (str): The path to the model file.
            load_replay_buffer (bool): Whether to load the replay buffer.
            fine_tuning (bool): If True, resets exploration rate for fine-tuning.
        """
        # Define the model parameters, ensuring consistency with __init__
        custom_objects = {
            "learning_rate": lambda p: 0.0001 + (0.001 - 0.0001) * p,  # Linear decay from 0.001 to 0.0001
            "buffer_size": 20000,
            "learning_starts": 2000,
            "batch_size": 64,
            "gamma": 0.95,
            "train_freq": (1, "step"),
            "gradient_steps": 1,
            "target_update_interval": 1800,
            "exploration_fraction": 0.1,
            "exploration_final_eps": 0.1,
            "exploration_initial_eps":1.0,
            "replay_buffer_class": RealActionReplayBuffer,
            "max_grad_norm": 10,
        }

        if fine_tuning:
            # For fine-tuning, we load the model but override the exploration
            # schedule to start from a low value.
            print("--- Loading model for fine-tuning, resetting exploration rate. ---")
            custom_objects["exploration_initial_eps"] = 0.1
        
        self.model = DQN.load(
            path, 
            env=self.model.get_env(), 
            custom_objects=custom_objects,
            device=self.model.device
        )
        
        if load_replay_buffer:
            replay_buffer_path = path.replace(".zip", "_replay_buffer.pkl")
            if os.path.exists(replay_buffer_path):
                self.model.load_replay_buffer(replay_buffer_path)
                print(f"--- Replay buffer loaded from {replay_buffer_path} ---")
            else:
                print("--- No replay buffer found, starting with a new one. ---")

    def save(self, path):
        """
        Saves the current model and its replay buffer.
        """
        self.model.save(path)
        
        replay_buffer_path = path.replace(".zip", "_replay_buffer.pkl")
        self.model.save_replay_buffer(replay_buffer_path)
        print(f"--- Replay buffer saved to {replay_buffer_path} ---")
