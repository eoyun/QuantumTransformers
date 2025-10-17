"""Quantum layers implemented with PennyLane for PyTorch models."""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import inspect

import pennylane as qml
import torch
from torch import nn


def angle_embedding(inputs: torch.Tensor, wires: Sequence[int]) -> None:
    """Applies angle embedding to the given ``inputs`` on ``wires``."""
    qml.templates.AngleEmbedding(inputs, wires=wires)


def basic_vqc(weights: torch.Tensor, wires: Sequence[int]) -> None:
    """Applies a basic entangling variational circuit."""
    qml.templates.BasicEntanglerLayers(weights, wires=wires)


def default_circuit(inputs: torch.Tensor, weights: torch.Tensor, wires: Sequence[int]) -> list:
    """Default circuit used by :class:`QuantumLayer`.

    It applies :func:`angle_embedding` followed by :func:`basic_vqc` and
    returns a list of ``Z`` expectation values, one per wire.
    """
    angle_embedding(inputs, wires=wires)
    basic_vqc(weights, wires=wires)
    return [qml.expval(qml.PauliZ(wire)) for wire in wires]


class QuantumLayer(nn.Module):
    """A PennyLane-backed quantum layer compatible with PyTorch modules."""

    def __init__(
        self,
        num_qubits: int,
        w_shape: tuple[int, ...] = (1,),
        circuit: Optional[Callable] = None,
        *,
        device_name: str = "default.qubit",
        shots: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.w_shape = w_shape
        self._circuit = circuit or default_circuit
        self._expects_wires = "wires" in inspect.signature(self._circuit).parameters

        self._device = qml.device(device_name, wires=num_qubits, shots=shots)

        def qnode_fn(inputs, weights):
            if self._expects_wires:
                return self._circuit(inputs, weights, wires=range(self.num_qubits))
            return self._circuit(inputs, weights)

        qnode = qml.QNode(qnode_fn, self._device, interface="torch", diff_method="best")
        weight_shapes = {"weights": w_shape + (num_qubits,)}
        self.layer = qml.qnn.TorchLayer(qnode, weight_shapes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        original_shape = x.shape
        x_flat = x.reshape(-1, original_shape[-1])
        outputs = self.layer(x_flat)
        outputs = outputs.to(dtype=x.dtype)
        outputs = outputs.reshape(*original_shape[:-1], outputs.shape[-1])
        return outputs


def get_circuit(
    embedding: Callable[[torch.Tensor, Sequence[int]], None] = angle_embedding,
    vqc: Callable[[torch.Tensor, Sequence[int]], None] = basic_vqc,
) -> Callable:
    """Returns a callable that can be used with :class:`QuantumLayer`.

    The returned callable expects three positional arguments ``(inputs, weights, wires)``
    and applies the provided ``embedding`` and ``vqc`` templates before measuring a list of
    ``Z`` expectation values, one per wire.
    """

    def circuit(inputs, weights, wires):
        embedding(inputs, wires=wires)
        vqc(weights, wires=wires)
        return [qml.expval(qml.PauliZ(wire)) for wire in wires]

    return circuit
