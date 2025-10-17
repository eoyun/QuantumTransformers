"""Training utilities implemented with PyTorch."""
from __future__ import annotations

import copy
import time
from typing import Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import auc, roc_auc_score, roc_curve
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}{bar:-10b}'


def _as_tensor(array, device: torch.device) -> torch.Tensor:
    if isinstance(array, torch.Tensor):
        tensor = array.to(device)
    else:
        tensor = torch.as_tensor(array, device=device)
    if tensor.dtype == torch.float64:
        tensor = tensor.float()
    return tensor


def _prepare_binary_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 1:
        return logits
    if logits.shape[-1] == 1:
        return logits.squeeze(-1)
    if logits.shape[-1] == 2:
        return logits[:, 1]
    raise ValueError("Binary classification expects logits with 1 or 2 outputs")


def _prepare_binary_targets(labels: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    target = labels.float()
    if target.shape != reference.shape:
        target = target.view(reference.shape)
    return target


def evaluate(
    model: nn.Module,
    dataloader,
    num_classes: int,
    device: torch.device,
    criterion: nn.Module,
    *,
    tqdm_desc: Optional[str] = None,
    debug: bool = False,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    was_training = model.training
    model.eval()

    losses = []
    logits_batches = []
    label_batches = []

    try:
        total = len(dataloader)
    except TypeError:
        total = None

    progress_bar = tqdm(
        total=total,
        desc=tqdm_desc,
        unit="batch",
        bar_format=TQDM_BAR_FORMAT,
        disable=tqdm_desc is None,
    )

    with torch.no_grad():
        for inputs_batch, labels_batch in dataloader:
            inputs = _as_tensor(inputs_batch, device)
            labels = _as_tensor(labels_batch, device)

            outputs = model(inputs)
            if num_classes == 2:
                logits_for_loss = _prepare_binary_logits(outputs)
                targets = _prepare_binary_targets(labels, logits_for_loss)
                loss = criterion(logits_for_loss, targets)
                logits_to_store = logits_for_loss
                labels_to_store = targets
            else:
                loss = criterion(outputs, labels.long())
                logits_to_store = outputs
                labels_to_store = labels

            losses.append(loss.item())
            logits_batches.append(logits_to_store.detach().cpu())
            label_batches.append(labels_to_store.detach().cpu())
            progress_bar.update(1)

    progress_bar.close()

    avg_loss = float(np.mean(losses)) if losses else 0.0
    logits = torch.cat(logits_batches) if logits_batches else torch.empty(0)
    labels = torch.cat(label_batches) if label_batches else torch.empty(0)

    if debug:
        print(f"logits = {logits}")

    if num_classes == 2 and logits.numel() > 0:
        probs = torch.sigmoid(logits).cpu().numpy()
        y_true = labels.float().view(-1).cpu().numpy()
        eval_fpr, eval_tpr, _ = roc_curve(y_true, probs)
        eval_auc = auc(eval_fpr, eval_tpr)
    elif num_classes > 2 and logits.numel() > 0:
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        y_true = labels.long().cpu().numpy()
        eval_auc = roc_auc_score(y_true, probs, multi_class='ovr')
        eval_fpr, eval_tpr = np.array([]), np.array([])
    else:
        eval_auc = 0.0
        eval_fpr, eval_tpr = np.array([]), np.array([])

    if debug:
        print(f"y_true = {labels}")

    if was_training:
        model.train()

    return avg_loss, eval_auc, eval_fpr, eval_tpr


def train_and_evaluate(
    model: nn.Module,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    num_classes: int,
    num_epochs: int,
    *,
    lrs_peak_value: float = 1e-3,
    lrs_warmup_steps: int = 5_000,
    lrs_decay_steps: int = 50_000,
    seed: int = 42,
    use_ray: bool = False,
    debug: bool = False,
    device: Optional[str] = None,
) -> dict:
    if use_ray:
        from ray.air import session

    torch.manual_seed(seed)
    np.random.seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)

    model = model.to(device_obj)

    num_parameters = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters = {num_parameters}")

    if num_classes == 2:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=lrs_peak_value)

    schedulers = []
    milestones = []
    if lrs_warmup_steps > 0:
        schedulers.append(LinearLR(optimizer, start_factor=0.0, end_factor=1.0, total_iters=lrs_warmup_steps))
        milestones.append(lrs_warmup_steps)
    schedulers.append(CosineAnnealingLR(optimizer, T_max=max(1, lrs_decay_steps), eta_min=0.0))
    scheduler = SequentialLR(optimizer, schedulers=schedulers, milestones=milestones) if len(schedulers) > 1 else schedulers[0]

    best_val_auc = 0.0
    best_epoch = 0
    best_state = None

    total_train_time = 0.0
    start_time = time.time()

    metrics = {
        'train_losses': [],
        'val_losses': [],
        'train_aucs': [],
        'val_aucs': [],
        'test_loss': 0.0,
        'test_auc': 0.0,
        'test_fpr': np.array([]),
        'test_tpr': np.array([]),
    }

    for epoch in range(num_epochs):
        model.train()
        try:
            train_total = len(train_dataloader)
        except TypeError:
            train_total = None

        progress_bar = tqdm(
            total=train_total,
            desc=f"Epoch {epoch + 1:3}/{num_epochs}",
            unit="batch",
            bar_format=TQDM_BAR_FORMAT,
        )
        epoch_train_time = time.time()

        for inputs_batch, labels_batch in train_dataloader:
            inputs = _as_tensor(inputs_batch, device_obj)
            labels = _as_tensor(labels_batch, device_obj)

            optimizer.zero_grad()
            outputs = model(inputs)

            if num_classes == 2:
                logits_for_loss = _prepare_binary_logits(outputs)
                targets = _prepare_binary_targets(labels, logits_for_loss)
                loss = criterion(logits_for_loss, targets)
            else:
                loss = criterion(outputs, labels.long())

            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            progress_bar.update(1)

        epoch_train_time = time.time() - epoch_train_time
        total_train_time += epoch_train_time

        train_loss, train_auc, _, _ = evaluate(
            model,
            train_dataloader,
            num_classes,
            device_obj,
            criterion,
            tqdm_desc=None,
            debug=debug,
        )
        val_loss, val_auc, _, _ = evaluate(
            model,
            val_dataloader,
            num_classes,
            device_obj,
            criterion,
            tqdm_desc=None,
            debug=debug,
        )

        progress_bar.set_postfix_str(
            f"Loss = {val_loss:.4f}, AUC = {val_auc:.3f}, Train time = {epoch_train_time:.2f}s"
        )
        progress_bar.close()

        metrics['train_losses'].append(train_loss)
        metrics['val_losses'].append(val_loss)
        metrics['train_aucs'].append(train_auc)
        metrics['val_aucs'].append(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())

        if use_ray:
            session.report({'val_loss': val_loss, 'val_auc': val_auc, 'best_val_auc': best_val_auc, 'best_epoch': best_epoch})

    metrics['train_losses'] = np.array(metrics['train_losses'])
    metrics['val_losses'] = np.array(metrics['val_losses'])
    metrics['train_aucs'] = np.array(metrics['train_aucs'])
    metrics['val_aucs'] = np.array(metrics['val_aucs'])

    print(f"Best validation AUC = {best_val_auc:.3f} at epoch {best_epoch}")
    print(f"Total training time = {total_train_time:.2f}s, total time (including evaluations) = {time.time() - start_time:.2f}s")

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_auc, test_fpr, test_tpr = evaluate(
        model,
        test_dataloader,
        num_classes,
        device_obj,
        criterion,
        tqdm_desc="Testing",
        debug=debug,
    )
    metrics['test_loss'] = test_loss
    metrics['test_auc'] = test_auc
    metrics['test_fpr'] = test_fpr
    metrics['test_tpr'] = test_tpr

    if use_ray:
        session.report({'test_loss': test_loss, 'test_auc': test_auc})

    return metrics
