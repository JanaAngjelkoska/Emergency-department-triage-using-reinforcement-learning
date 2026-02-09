import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int = 64):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
