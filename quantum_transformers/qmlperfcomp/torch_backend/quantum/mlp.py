"""Quantum-enhanced multilayer perceptron implemented with PyTorch."""

import torch


def MLP(
    hidden_size: int,
    qml_backend: str = "pennylane",
    qdevice: str = "default.qubit",
    qdiff_method: str = "best",
) -> torch.nn.Module:
    if qml_backend == "pennylane":
        from quantum_transformers.qmlperfcomp.torch_backend.quantum.pennylane_backend import (
            QuantumLayer,
        )
    elif qml_backend == "tensorcircuit":
        from quantum_transformers.qmlperfcomp.torch_backend.quantum.tensorcircuit_backend import (
            QuantumLayer,
        )
    else:
        raise ValueError(f"Unknown qml_backend: {qml_backend}")

    class QuantumMLP(torch.nn.Module):
        def __init__(self, hidden_size: int, qdevice: str, qdiff_method: str) -> None:
            super().__init__()
            self.fc1 = torch.nn.Sequential(torch.nn.LazyLinear(hidden_size), torch.nn.ReLU())
            self.fc2 = torch.nn.Sequential(
                QuantumLayer(hidden_size, qdevice=qdevice, diff_method=qdiff_method),
                torch.nn.ReLU(),
            )
            self.fc3 = torch.nn.Linear(hidden_size, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)
            return x

    return QuantumMLP(hidden_size, qdevice=qdevice, qdiff_method=qdiff_method)
