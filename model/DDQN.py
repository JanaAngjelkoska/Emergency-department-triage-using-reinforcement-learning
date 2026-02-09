import random

import numpy as np
import torch
from model.DQN import DQN


class DDQN(DQN):
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
                    next_q_values = self.model(next_state_tensor)
                    best_action = torch.argmax(next_q_values).item()

                    target_q_values = self.target_model(next_state_tensor)
                    max_q = target_q_values[0][best_action].item()

                    target_q = reward + self.discount_factor * max_q

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
