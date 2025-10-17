import os
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import gdown
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as T
from torchvision import datasets as tv_datasets

_PAD_TOKEN = "[PAD]"
_UNK_TOKEN = "[UNK]"
_START_TOKEN = "[START]"
_END_TOKEN = "[END]"
_IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
_IMDB_ARCHIVE = "aclImdb_v1.tar.gz"
_IMDB_DIR = "aclImdb"


class _ChannelLastWrapper(Dataset):
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.dataset)

    def __getitem__(self, idx):  # type: ignore[override]
        image, label = self.dataset[idx]
        if isinstance(image, torch.Tensor):
            if image.ndim == 3:
                image = image.permute(1, 2, 0)
        elif isinstance(image, np.ndarray):
            tensor = torch.from_numpy(image)
            if tensor.ndim == 3:
                tensor = tensor.permute(1, 2, 0)
            image = tensor
        else:
            image = torch.as_tensor(image)
            if image.ndim == 3:
                image = image.permute(1, 2, 0)
        return image, label


class _NpyImageDataset(Dataset):
    def __init__(self, samples: Sequence[tuple[Path, int]], img_shape: tuple[int, int, int]) -> None:
        self.samples = list(samples)
        self.img_shape = img_shape

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.samples)

    def __getitem__(self, idx):  # type: ignore[override]
        path, label = self.samples[idx]
        image = np.load(path).astype(np.float32)
        expected = self.img_shape
        if image.shape != expected:
            if image.ndim == 3 and image.shape[0] == expected[-1] and image.shape[1:] == expected[:2]:
                image = np.transpose(image, (1, 2, 0))
            elif image.ndim == 3 and image.shape[-1] == expected[0] and image.shape[:2] == expected[1:]:
                image = np.transpose(image, (1, 0, 2))
            else:
                raise ValueError(f"Unexpected image shape {image.shape} for {path}")
        tensor = torch.from_numpy(image)
        return tensor, int(label)


class _IMDBTokenizer:
    def __call__(self, text: str) -> list[str]:
        return _tokenize(text)


class _IMDBDataset(Dataset):
    def __init__(
        self,
        examples: Sequence[tuple[str, int]],
        vocab_lookup: dict[str, int],
        max_seq_len: int,
    ) -> None:
        self.examples = list(examples)
        self.vocab_lookup = vocab_lookup
        self.max_seq_len = max_seq_len
        self.pad_idx = vocab_lookup[_PAD_TOKEN]
        self.unk_idx = vocab_lookup[_UNK_TOKEN]
        self.start_idx = vocab_lookup[_START_TOKEN]
        self.end_idx = vocab_lookup[_END_TOKEN]

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.examples)

    def __getitem__(self, idx):  # type: ignore[override]
        text, label = self.examples[idx]
        tokens = _tokenize(text)
        token_ids = [self.start_idx]
        for token in tokens:
            token_ids.append(self.vocab_lookup.get(token, self.unk_idx))
            if len(token_ids) >= self.max_seq_len - 1:
                break
        token_ids.append(self.end_idx)
        if len(token_ids) < self.max_seq_len:
            token_ids.extend([self.pad_idx] * (self.max_seq_len - len(token_ids)))
        else:
            token_ids = token_ids[:self.max_seq_len]
        input_tensor = torch.tensor(token_ids, dtype=torch.long)
        return input_tensor, int(label)


def _tokenize(text: str) -> list[str]:
    tokens = []
    current = []
    for char in text.lower():
        if char.isalnum():
            current.append(char)
        else:
            if current:
                tokens.append("".join(current))
                current.clear()
    if current:
        tokens.append("".join(current))
    return tokens


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    with urllib.request.urlopen(url) as response, open(destination, "wb") as output:
        output.write(response.read())


def _ensure_imdb_dataset(root: Path) -> Path:
    dataset_dir = root / _IMDB_DIR
    if dataset_dir.exists():
        return dataset_dir
    archive_path = root / _IMDB_ARCHIVE
    print(f"Downloading IMDB dataset to {archive_path}")
    _download_file(_IMDB_URL, archive_path)
    print(f"Extracting {archive_path}")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=root)
    archive_path.unlink(missing_ok=True)
    return dataset_dir


def _load_imdb_split(split_dir: Path) -> list[tuple[str, int]]:
    examples: list[tuple[str, int]] = []
    for label_name, label_value in ("pos", 1), ("neg", 0):
        label_dir = split_dir / label_name
        for path in sorted(label_dir.glob("*.txt")):
            text = path.read_text(encoding="utf-8", errors="ignore")
            examples.append((text, label_value))
    return examples


def _build_imdb_vocab(examples: Iterable[tuple[str, int]], max_vocab_size: int) -> list[str]:
    counter: Counter[str] = Counter()
    for text, _ in examples:
        counter.update(_tokenize(text))
    reserved = [_PAD_TOKEN, _UNK_TOKEN, _START_TOKEN, _END_TOKEN]
    most_common = [token for token, _ in counter.most_common(max(0, max_vocab_size - len(reserved)))]
    return reserved + most_common


def _numpy_folder_samples(dataset_path: Path) -> list[tuple[Path, int]]:
    samples: list[tuple[Path, int]] = []
    class_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
    class_map = {directory.name: index for index, directory in enumerate(class_dirs)}
    for class_dir in class_dirs:
        for path in sorted(class_dir.glob("*.npy")):
            samples.append((path, class_map[class_dir.name]))
    return samples


def _split_samples(samples: list[tuple[Path, int]], val_fraction: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(samples))
    rng.shuffle(indices)
    split_idx = int(len(samples) * (1 - val_fraction))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    train_samples = [samples[i] for i in train_indices]
    val_samples = [samples[i] for i in val_indices]
    return train_samples, val_samples


def _make_dataloader(dataset: Dataset, batch_size: int, *, shuffle: bool, drop_last: bool):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


def get_mnist_dataloaders(data_dir: str = "~/data", batch_size: int = 1, drop_remainder: bool = True):
    data_dir = os.path.expanduser(data_dir)
    transform = T.Compose([
        T.ToTensor(),
    ])
    train_dataset = tv_datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = tv_datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    train_len = int(len(train_dataset) * 0.9)
    val_len = len(train_dataset) - train_len
    train_subset, val_subset = random_split(
        train_dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(0),
    )
    train_wrapper = _ChannelLastWrapper(train_subset)
    val_wrapper = _ChannelLastWrapper(val_subset)
    test_wrapper = _ChannelLastWrapper(test_dataset)
    train_loader = _make_dataloader(train_wrapper, batch_size, shuffle=True, drop_last=drop_remainder)
    val_loader = _make_dataloader(val_wrapper, batch_size, shuffle=False, drop_last=drop_remainder)
    test_loader = _make_dataloader(test_wrapper, batch_size, shuffle=False, drop_last=drop_remainder)
    return train_loader, val_loader, test_loader


def _ensure_numpy_dataset(data_dir: Path, name: str, gdrive_id: str) -> Path:
    dataset_dir = data_dir / name
    if dataset_dir.exists():
        return dataset_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    archive_path = data_dir / f"{name}.tar.xz"
    print(f"Downloading {name} dataset")
    gdown.download(id=gdrive_id, output=str(archive_path), quiet=False)
    print(f"Extracting {archive_path}")
    with tarfile.open(archive_path, "r:xz") as tar:
        tar.extractall(path=data_dir)
    archive_path.unlink(missing_ok=True)
    return dataset_dir


def _numpy_dataloaders(
    dataset_dir: Path,
    *,
    img_shape: tuple[int, int, int],
    batch_size: int,
    drop_remainder: bool,
    val_fraction: float = 0.1,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_samples = _numpy_folder_samples(dataset_dir / "train")
    test_samples = _numpy_folder_samples(dataset_dir / "test")
    train_split, val_split = _split_samples(train_samples, val_fraction=val_fraction)
    train_dataset = _NpyImageDataset(train_split, img_shape)
    val_dataset = _NpyImageDataset(val_split, img_shape)
    test_dataset = _NpyImageDataset(test_samples, img_shape)
    train_loader = _make_dataloader(train_dataset, batch_size, shuffle=True, drop_last=drop_remainder)
    val_loader = _make_dataloader(val_dataset, batch_size, shuffle=False, drop_last=drop_remainder)
    test_loader = _make_dataloader(test_dataset, batch_size, shuffle=False, drop_last=drop_remainder)
    return train_loader, val_loader, test_loader


def get_electron_photon_dataloaders(
    data_dir: str = "~/data",
    batch_size: int = 1,
    drop_remainder: bool = True,
):
    data_path = Path(os.path.expanduser(data_dir))
    dataset_dir = _ensure_numpy_dataset(
        data_path,
        name="electron-photon",
        gdrive_id="1VAqGQaMS5jSWV8gTXw39Opz-fNMsDZ8e",
    )
    return _numpy_dataloaders(
        dataset_dir,
        img_shape=(32, 32, 2),
        batch_size=batch_size,
        drop_remainder=drop_remainder,
    )


def get_quark_gluon_dataloaders(
    data_dir: str = "~/data",
    batch_size: int = 1,
    drop_remainder: bool = True,
):
    data_path = Path(os.path.expanduser(data_dir))
    dataset_dir = _ensure_numpy_dataset(
        data_path,
        name="quark-gluon",
        gdrive_id="1PL2YEr5V__zUZVuUfGdUvFTkE9ULHayz",
    )
    return _numpy_dataloaders(
        dataset_dir,
        img_shape=(125, 125, 3),
        batch_size=batch_size,
        drop_remainder=drop_remainder,
    )


def get_imdb_dataloaders(
    data_dir: str = "~/data",
    batch_size: int = 1,
    drop_remainder: bool = True,
    max_vocab_size: int = 20_000,
    max_seq_len: int = 512,
):
    data_path = Path(os.path.expanduser(data_dir))
    dataset_dir = _ensure_imdb_dataset(data_path)
    train_examples = _load_imdb_split(dataset_dir / "train")
    test_examples = _load_imdb_split(dataset_dir / "test")
    train_len = int(len(train_examples) * 0.9)
    val_len = len(train_examples) - train_len
    train_split = train_examples[:train_len]
    val_split = train_examples[train_len:]
    vocab = _build_imdb_vocab(train_split, max_vocab_size=max_vocab_size)
    vocab_lookup = {token: idx for idx, token in enumerate(vocab)}
    train_dataset = _IMDBDataset(train_split, vocab_lookup, max_seq_len)
    val_dataset = _IMDBDataset(val_split, vocab_lookup, max_seq_len)
    test_dataset = _IMDBDataset(test_examples, vocab_lookup, max_seq_len)
    train_loader = _make_dataloader(train_dataset, batch_size, shuffle=True, drop_last=drop_remainder)
    val_loader = _make_dataloader(val_dataset, batch_size, shuffle=False, drop_last=drop_remainder)
    test_loader = _make_dataloader(test_dataset, batch_size, shuffle=False, drop_last=drop_remainder)
    tokenizer = _IMDBTokenizer()
    return (train_loader, val_loader, test_loader), vocab, tokenizer
