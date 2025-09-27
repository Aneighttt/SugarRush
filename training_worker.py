import torch
import time
from config import MODEL_PATH

def training_worker(model, trajectory_queue):
    """
    This function runs in a background thread and continuously trains the shared PPO model.

    It works by pulling completed trajectories from a shared queue. Each trajectory is a
    list of experiences (observation, action, reward, done) from a single agent's episode
    or a collected batch.

    The process is as follows:
    1. Wait for a trajectory to become available in `trajectory_queue`.
    2. Add all experiences from the trajectory into the model's `rollout_buffer`.
    3. Once the data is added, trigger the model's `train()` method to perform a policy update.
    4. After training, the rollout buffer is reset to be ready for the next batch of experiences.
    5. The updated model is saved to disk.
    """
    print("Training worker started...")
    while True:
        # This will block until a trajectory is available
        print("try get")
        start_time = time.time()
        trajectory = trajectory_queue.get()
        
        # A way to gracefully shut down the worker if needed
        if trajectory is None:
            break
        
        print(f"Training worker: Received a trajectory of length {len(trajectory)}. Starting training...")
        
        # Add collected experiences to the model's buffer
        for experience in trajectory:
            obs, action, reward, done, info, value, log_prob = experience
            
            # obs and action are numpy arrays, value and log_prob are CPU tensors.
            # The buffer handles the conversion internally.
            model.rollout_buffer.add(
                obs,
                action,
                reward,
                done,
                value,
                log_prob
            )
        print("finish add")
        # Perform a training step
        model.learn(total_timesteps = 1800*512)
        
        # Clear the buffer for the next batch of data
        model.rollout_buffer.reset()
        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        print("Training worker: Training complete. Saving model...,  {elapsed_ms:.2f}")
