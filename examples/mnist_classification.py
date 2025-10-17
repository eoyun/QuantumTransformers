"""MNIST classification example using the QuantumTransformers library."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Optional

import torch

from quantum_transformers.datasets import get_mnist_dataloaders
from quantum_transformers.training import train_and_evaluate
from quantum_transformers.transformers import VisionTransformer


@dataclass
class TrainingConfig:
    data_dir: str = "~/data"
    batch_size: int = 128
    num_epochs: int = 5
    learning_rate: float = 3e-4
    warmup_steps: int = 100
    decay_steps: int = 1000
    seed: int = 42
    device: Optional[str] = None


def build_model() -> VisionTransformer:
    """Create a small Vision Transformer suited for 28x28 MNIST images."""
    return VisionTransformer(
        num_classes=10,
        patch_size=7,
        hidden_size=128,
        num_heads=4,
        num_transformer_blocks=4,
        mlp_hidden_size=256,
        dropout=0.1,
        pos_embedding="sincos",
        classifier="gap",
        channels_last=True,
    )


def run_training(config: TrainingConfig) -> None:
    print("Loading MNIST dataloaders …")
    train_loader, val_loader, test_loader = get_mnist_dataloaders(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        drop_remainder=False,
    )

    model = build_model()

    print("Starting training …")
    metrics = train_and_evaluate(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        num_classes=10,
        num_epochs=config.num_epochs,
        lrs_peak_value=config.learning_rate,
        lrs_warmup_steps=config.warmup_steps,
        lrs_decay_steps=config.decay_steps,
        seed=config.seed,
        device=config.device,
    )

    test_loss = metrics["test_loss"]
    test_auc_raw = metrics["test_auc"]
    try:
        test_auc = float(test_auc_raw)
    except (TypeError, ValueError):
        test_auc = math.nan

    test_accuracy = compute_accuracy(model, test_loader)

    print(f"Test loss: {test_loss:.4f}")
    if math.isnan(test_auc):
        print("Test AUC: NaN (not enough samples for multi-class AUC computation)")
    else:
        print(f"Test AUC: {test_auc:.4f}")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")


def compute_accuracy(model: VisionTransformer, dataloader) -> float:
    device = next(model.parameters()).device
    param_dtype = next(model.parameters()).dtype
    was_training = model.training
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device=device, dtype=param_dtype)
            labels = labels.to(device=device)
            logits = model(inputs)
            predictions = logits.argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.numel()

    if was_training:
        model.train()

    return correct / total if total else 0.0


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="~/data", help="Directory to download/store MNIST")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Linear warmup steps")
    parser.add_argument("--decay-steps", type=int, default=1000, help="Cosine annealing steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        default=None,
        help="PyTorch device string (e.g. 'cuda', 'cuda:0', or 'cpu'). Defaults to auto-detect.",
    )
    args = parser.parse_args()
    return TrainingConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        decay_steps=args.decay_steps,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    run_training(parse_args())
