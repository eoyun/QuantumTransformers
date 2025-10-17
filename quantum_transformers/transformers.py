"""Transformer architectures implemented with PyTorch."""
from __future__ import annotations

import math
from typing import Callable, Literal, Optional

import torch
from torch import nn
from torch.nn import functional as F

from .quantum_layer import QuantumLayer


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
        *,
        quantum_w_shape: tuple[int, ...] = (1,),
        quantum_circuit: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        if quantum_circuit is None:
            self.q_proj = nn.Linear(hidden_size, hidden_size)
            self.k_proj = nn.Linear(hidden_size, hidden_size)
            self.v_proj = nn.Linear(hidden_size, hidden_size)
            self.out_proj = nn.Linear(hidden_size, hidden_size)
        else:
            self.q_proj = QuantumLayer(hidden_size, w_shape=quantum_w_shape, circuit=quantum_circuit)
            self.k_proj = QuantumLayer(hidden_size, w_shape=quantum_w_shape, circuit=quantum_circuit)
            self.v_proj = QuantumLayer(hidden_size, w_shape=quantum_w_shape, circuit=quantum_circuit)
            self.out_proj = QuantumLayer(hidden_size, w_shape=quantum_w_shape, circuit=quantum_circuit)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn_logits, dim=-1)
        attn = self.attn_dropout(attn)

        values = torch.matmul(attn, v)
        values = values.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.out_proj(values)
        output = self.out_dropout(output)
        return output


class FeedForward(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        mlp_hidden_size: int,
        dropout: float = 0.0,
        *,
        quantum_w_shape: tuple[int, ...] = (1,),
        quantum_circuit: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, mlp_hidden_size)
        self.quantum_layer = (
            QuantumLayer(mlp_hidden_size, w_shape=quantum_w_shape, circuit=quantum_circuit)
            if quantum_circuit is not None
            else None
        )
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(mlp_hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.fc1(x)
        if self.quantum_layer is not None:
            x = self.quantum_layer(x)
        x = self.dropout(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_hidden_size: int,
        dropout: float = 0.0,
        *,
        quantum_w_shape: tuple[int, ...] = (1,),
        quantum_attn_circuit: Optional[Callable] = None,
        quantum_mlp_circuit: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = MultiHeadSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            quantum_w_shape=quantum_w_shape,
            quantum_circuit=quantum_attn_circuit,
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = FeedForward(
            hidden_size=hidden_size,
            mlp_hidden_size=mlp_hidden_size,
            dropout=dropout,
            quantum_w_shape=quantum_w_shape,
            quantum_circuit=quantum_mlp_circuit,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        attn_output = self.attn(self.norm1(x))
        x = x + self.dropout1(attn_output)
        mlp_output = self.mlp(self.norm2(x))
        x = x + self.dropout2(mlp_output)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        max_seq_len: int,
        num_classes: int,
        hidden_size: int,
        num_heads: int,
        num_transformer_blocks: int,
        mlp_hidden_size: int,
        dropout: float = 0.0,
        *,
        quantum_w_shape: tuple[int, ...] = (1,),
        quantum_attn_circuit: Optional[Callable] = None,
        quantum_mlp_circuit: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(num_embeddings=num_tokens, embedding_dim=hidden_size)
        self.position_embedding = nn.Embedding(num_embeddings=max_seq_len, embedding_dim=hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_hidden_size=mlp_hidden_size,
                    dropout=dropout,
                    quantum_w_shape=quantum_w_shape,
                    quantum_attn_circuit=quantum_attn_circuit,
                    quantum_mlp_circuit=quantum_mlp_circuit,
                )
                for _ in range(num_transformer_blocks)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        positions = torch.arange(x.size(1), device=x.device)
        x = self.token_embedding(x) + self.position_embedding(positions)[None, :, :]
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        logits = self.classifier(x)
        return logits


def posemb_sincos_2d(
    sqrt_num_steps: int,
    hidden_size: int,
    *,
    temperature: float = 10_000.0,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """2D sine-cosine positional embedding adapted for PyTorch."""
    if hidden_size % 4 != 0:
        raise ValueError("hidden_size must be divisible by 4 for sin-cos position embedding")

    y, x = torch.meshgrid(
        torch.arange(sqrt_num_steps, device=device, dtype=dtype),
        torch.arange(sqrt_num_steps, device=device, dtype=dtype),
        indexing="ij",
    )
    omega = torch.arange(hidden_size // 4, device=device, dtype=dtype)
    if omega.numel() > 1:
        omega = omega / (omega.numel() - 1)
    omega = 1.0 / (temperature ** omega)

    y = torch.einsum("m,d->md", y.reshape(-1), omega)
    x = torch.einsum("m,d->md", x.reshape(-1), omega)
    pe = torch.cat([torch.sin(x), torch.cos(x), torch.sin(y), torch.cos(y)], dim=1)
    return pe.unsqueeze(0)


class VisionTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        patch_size: int,
        hidden_size: int,
        num_heads: int,
        num_transformer_blocks: int,
        mlp_hidden_size: int,
        dropout: float = 0.1,
        *,
        pos_embedding: Literal["none", "learn", "sincos"] = "learn",
        classifier: Literal["token", "gap"] = "gap",
        channels_last: bool = True,
        quantum_w_shape: tuple[int, ...] = (1,),
        quantum_attn_circuit: Optional[Callable] = None,
        quantum_mlp_circuit: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.pos_embedding_type = pos_embedding
        self.classifier_type = classifier
        self.channels_last = channels_last

        self.patch_embed: Optional[nn.Conv2d] = None
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_hidden_size=mlp_hidden_size,
                    dropout=dropout,
                    quantum_w_shape=quantum_w_shape,
                    quantum_attn_circuit=quantum_attn_circuit,
                    quantum_mlp_circuit=quantum_mlp_circuit,
                )
                for _ in range(num_transformer_blocks)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, num_classes)

        if self.pos_embedding_type == "learn":
            self.register_parameter("pos_embedding", None)
        else:
            self.register_buffer("pos_embedding", None, persistent=False)

        if self.classifier_type == "token":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        else:
            self.register_parameter("cls_token", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.channels_last:
            x = x.permute(0, 3, 1, 2)
        batch_size, channels, height, width = x.shape
        if height != width:
            raise ValueError(f"Input must be square, got {height}x{width}")
        if height % self.patch_size != 0:
            raise ValueError("Image size must be divisible by patch_size")

        if self.patch_embed is None or self.patch_embed.in_channels != channels:
            conv = nn.Conv2d(
                in_channels=channels,
                out_channels=self.hidden_size,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            )
            conv = conv.to(device=x.device, dtype=x.dtype)
            self.patch_embed = conv
        else:
            self.patch_embed = self.patch_embed.to(device=x.device, dtype=x.dtype)
        x = self.patch_embed(x)
        sqrt_num_steps = x.shape[2]
        x = x.flatten(2).transpose(1, 2)
        num_steps = x.size(1)

        if self.pos_embedding_type == "learn":
            if self.pos_embedding is None or self.pos_embedding.size(1) != num_steps:
                pos = torch.randn(1, num_steps, self.hidden_size, device=x.device, dtype=x.dtype)
                pos /= math.sqrt(self.hidden_size)
                self.pos_embedding = nn.Parameter(pos)
            x = x + self.pos_embedding
        elif self.pos_embedding_type == "sincos":
            pos = posemb_sincos_2d(sqrt_num_steps, self.hidden_size, dtype=x.dtype, device=x.device)
            x = x + pos
        elif self.pos_embedding_type == "none":
            pass
        else:
            raise ValueError(f"Unknown positional embedding type: {self.pos_embedding_type}")

        if self.classifier_type == "token":
            cls_token = self.cls_token.to(device=x.device, dtype=x.dtype)
            cls_token = cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_token, x], dim=1)

        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        if self.classifier_type == "token":
            x = x[:, 0]
        elif self.classifier_type == "gap":
            x = x.mean(dim=1)
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")

        logits = self.head(x)
        return logits
