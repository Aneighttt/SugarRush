import torch
from stable_baselines3 import DQN
from environment import BomberEnv

class SB3_DQNAgent:
    def __init__(self, env: BomberEnv, log_dir="./log/"):
        """
        Initializes a DQNAgent that wraps a Stable Baselines 3 DQN model.
        The agent is now tightly coupled with a Gym environment.

        Args:
            env (BomberEnv): The custom Gym environment.
            log_dir (str): Directory to save tensorboard logs.
        """
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        self.model = DQN(
            policy="MultiInputPolicy",
            env=env,
            learning_rate=0.001,
            buffer_size=20000,
            learning_starts=100, # Start training after 100 steps
            batch_size=32,
            gamma=0.95,
            train_freq=(1, "step"),
            gradient_steps=1,
            target_update_interval=100,
            exploration_fraction=0.5,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            verbose=1,
            tensorboard_log=log_dir,
            device=device
        )

    def choose_action(self, observation):
        """
        Chooses an action using the model's prediction.
        """
        action, _states = self.model.predict(observation, deterministic=False)
        return action.item()

    def load(self, path):
        """
        Loads a pre-trained model.
        """
        self.model = DQN.load(path, env=self.model.get_env(), device=self.model.device)

    def save(self, path):
        """
        Saves the current model.
        """
        self.model.save(path)
