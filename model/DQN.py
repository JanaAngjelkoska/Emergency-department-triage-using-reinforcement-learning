from collections import deque
import random
from typing import Tuple, List, Optional

import numpy as np
import torch
from torch import optim, nn

from model.QNetwork import QNetwork


class DQN:
    def __init__(self,
                 state_space_shape: int,
                 num_actions: int,
                 model: QNetwork,
                 target_model: QNetwork,
                 learning_rate: float = 0.001,
                 discount_factor: float = 0.95,
                 batch_size: int = 32,
                 memory_size: int = 10000):

        self.state_space_shape = state_space_shape
        self.num_actions = num_actions
        self.discount_factor = discount_factor
        self.batch_size = batch_size

        self.memory = deque(maxlen=memory_size)

        self.model = model
        self.target_model = target_model

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.update_target_model()

        self.train_step_count = 0
        self.recent_losses = deque(maxlen=100)

    def update_memory(self,
                      state: np.ndarray,
                      action: int,
                      reward: float,
                      next_state: np.ndarray,
                      done: bool) -> None:
        normalized_reward = reward / 100.0
        self.memory.append((state, action, normalized_reward, next_state, done))

    def update_target_model(self) -> None:
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, state: np.ndarray, epsilon: float) -> int:
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.num_actions)

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()

    def train(self) -> None:
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        states = []
        actions = []
        targets = []

        for state, action, reward, next_state, done in minibatch:
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            if done:
                target_q = reward
            else:
                with torch.no_grad():
                    next_q_values = self.target_model(next_state_tensor)
                    max_future_q = torch.max(next_q_values).item()
                    target_q = reward + self.discount_factor * max_future_q

            states.append(state)
            actions.append(action)
            targets.append(target_q)

        states_tensor = torch.tensor(np.array(states), dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.long)
        targets_tensor = torch.tensor(targets, dtype=torch.float32)

        self.optimizer.zero_grad()

        q_values = self.model(states_tensor)
        q_values_for_actions = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze()

        loss = self.criterion(q_values_for_actions, targets_tensor)
        loss.backward()
        self.optimizer.step()

        self.recent_losses.append(loss.item())
        self.train_step_count += 1

        if self.train_step_count % 10000 == 0:
            avg_loss = np.mean(self.recent_losses)
            print(f"[Train step {self.train_step_count}] Avg loss (last 100): {avg_loss:.4f}, Q-value range: [{q_values.min().item():.1f}, {q_values.max().item():.1f}]")

    def save(self, model_name: str, episode: int) -> None:
        torch.save(self.model.state_dict(), f'dqn_{model_name}_{episode}.pt')

    def load(self, model_name: str, episode: int) -> None:
        self.model.load_state_dict(torch.load(f'dqn_{model_name}_{episode}.pt'))