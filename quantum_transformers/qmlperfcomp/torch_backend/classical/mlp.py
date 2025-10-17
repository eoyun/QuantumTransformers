"""Classical multilayer perceptron implemented in PyTorch."""

import torch


class MLP(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.fc1 = torch.nn.Sequential(torch.nn.LazyLinear(hidden_size), torch.nn.ReLU())
        self.fc2 = torch.nn.Sequential(torch.nn.Linear(hidden_size, hidden_size), torch.nn.ReLU())
        self.fc3 = torch.nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
