import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from model import DQN

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.update_target_every = 100 # Ticks to update target network
        self.train_counter = 0
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = self._build_model().to(self.device)
        self.target_model = self._build_model().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.update_target_model()

    def _build_model(self):
        return DQN(self.state_size, self.action_size)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state = state.to(self.device)
            act_values = self.model(state)
            return torch.argmax(act_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        
        states = torch.cat([s[0] for s in minibatch]).to(self.device)
        actions = torch.tensor([s[1] for s in minibatch]).to(self.device)
        rewards = torch.tensor([s[2] for s in minibatch]).to(self.device)
        next_states = torch.cat([s[3] for s in minibatch]).to(self.device)
        dones = torch.tensor([s[4] for s in minibatch]).to(self.device)

        # Get Q-values for current states from the main model
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))

        # Get max Q-values for next states from the target model
        next_q_values = self.target_model(next_states).max(1)[0].detach()

        # Compute the expected Q-values
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones.float()))

        # Compute loss
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_counter += 1
        if self.train_counter % self.update_target_every == 0:
            self.update_target_model()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name, map_location=self.device))
        self.update_target_model()

    def save(self, name):
        torch.save(self.model.state_dict(), name)
