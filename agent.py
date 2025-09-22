import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from model import DQN

class DQNAgent:
    def __init__(self, state_size, action_size, vector_size):
        self.state_size = state_size
        self.action_size = action_size
        self.vector_size = vector_size
        self.memory = deque(maxlen=20000)
        
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        # This decay rate is calculated to bring epsilon from 1.0 to ~0.05 in ~18000 steps (10 games).
        # ln(0.05) / 18000 = -0.000166 -> e^-0.000166 ~= 0.99983
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001
        self.update_target_every = 100 # Ticks to update target network
        self.train_counter = 0
        # We reset epsilon every 20 games (~36000 steps).
        # This provides a 10-game "learning phase" and a 10-game "exploitation phase".
        self.epsilon_reset_tick = 7200
        
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        self.model = self._build_model().to(self.device)
        self.target_model = self._build_model().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.update_target_model()

    def _build_model(self):
        return DQN(self.state_size, self.action_size, self.vector_size)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, visual_state, vector_state, action, reward, next_visual_state, next_vector_state, done):
        self.memory.append((visual_state, vector_state, action, reward, next_visual_state, next_vector_state, done))

    def choose_action(self, visual_state, vector_state):
        if np.random.rand() <= self.epsilon:
            # If action is random, there are no Q-values to return
            return random.randrange(self.action_size), None
        
        with torch.no_grad():
            visual_state = visual_state.to(self.device)
            vector_state = vector_state.to(self.device)
            act_values = self.model(visual_state, vector_state)
            # Return the best action and the Q-values for all actions
            return torch.argmax(act_values).item(), act_values

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return None
            
        minibatch = random.sample(self.memory, batch_size)
        
        visual_states = torch.cat([s[0] for s in minibatch]).to(self.device)
        vector_states = torch.cat([s[1] for s in minibatch]).to(self.device)
        actions = torch.tensor([s[2] for s in minibatch]).to(self.device)
        rewards = torch.tensor([s[3] for s in minibatch]).to(self.device)
        next_visual_states = torch.cat([s[4] for s in minibatch]).to(self.device)
        next_vector_states = torch.cat([s[5] for s in minibatch]).to(self.device)
        dones = torch.tensor([s[6] for s in minibatch]).to(self.device)

        # Get Q-values for current states from the main model
        current_q_values = self.model(visual_states, vector_states).gather(1, actions.unsqueeze(1))

        # Get max Q-values for next states from the target model
        next_q_values = self.target_model(next_visual_states, next_vector_states).max(1)[0].detach()

        # Compute the expected Q-values
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones.float()))

        # Compute loss
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # --- CRITICAL: Gradient Clipping to prevent exploding gradients ---
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self.train_counter += 1
        if self.train_counter % self.update_target_every == 0:
            self.update_target_model()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # --- Exploration Restart Logic ---
        if self.train_counter > 0 and self.train_counter % self.epsilon_reset_tick == 0:
            self.epsilon = 1.0
            print(f"--- EXPLORATION RESTART: Epsilon reset to 1.0 at training step {self.train_counter} ---")

        return loss.item()

    def load(self, name):
        self.model.load_state_dict(torch.load(name, map_location=self.device))
        self.update_target_model()

    def save(self, name):
        torch.save(self.model.state_dict(), name)
