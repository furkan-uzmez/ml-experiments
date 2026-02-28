"""Dataset and DataLoader utilities for gradient accumulation experiments."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


DEFAULT_CLASSES: Tuple[str, str] = ("AP", "PA")
DEFAULT_CLASS_TO_IDX: Dict[str, int] = {"AP": 0, "PA": 1}
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class DataLoaderConfig:
    """Configuration for creating optimized PyTorch DataLoaders.

    Attributes:
        batch_size: Number of samples per mini-batch.
        shuffle: Whether to shuffle dataset order.
        num_workers: Worker process count for asynchronous loading.
        pin_memory: Whether to pin host memory for faster CUDA transfers.
        drop_last: Drop incomplete tail batch.
        prefetch_factor: Number of prefetched batches per worker.
        persistent_workers: Keep workers alive across epochs.
    """

    batch_size: int = 32
    shuffle: bool = False
    num_workers: Optional[int] = None
    pin_memory: Optional[bool] = None
    drop_last: bool = False
    prefetch_factor: int = 2
    persistent_workers: bool = True


def build_transforms(
    image_size: int = 224,
    augment: bool = False,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> transforms.Compose:
    """Create a default image preprocessing pipeline.

    Args:
        image_size: Target square image size.
        augment: Whether to add lightweight augmentation.
        mean: Channel-wise normalization means.
        std: Channel-wise normalization standard deviations.

    Returns:
        A torchvision transform pipeline.
    """

    ops: list[Any] = [transforms.Resize((image_size, image_size))]

    if augment:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),
            ]
        )

    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return transforms.Compose(ops)


class COVIDCXNetDataset(Dataset):
    """Dataset wrapper for AP/PA projection classification.

    The class reads metadata from a CSV file and resolves file paths relative
    to `root_dir`. It supports split filtering and optional image caching.
    """

    def __init__(
        self,
        csv_file: str | os.PathLike[str],
        root_dir: str | os.PathLike[str],
        transform: Optional[Any] = None,
        split: str = "all",
        cache_images: bool = False,
        image_mode: str = "RGB",
        split_column: str = "split",
        path_column: str = "filepath",
        label_column: str = "projection",
        classes: Iterable[str] = DEFAULT_CLASSES,
    ) -> None:
        """Initialize the dataset.

        Args:
            csv_file: Metadata CSV path.
            root_dir: Root directory used to resolve image file paths.
            transform: Transform callable applied to each image.
            split: Dataset split to filter (`train`, `val`, `test`, or `all`).
            cache_images: Whether to cache decoded PIL images in memory.
            image_mode: PIL conversion mode (default: `RGB`).
            split_column: Column name storing split labels.
            path_column: Column name storing relative image path.
            label_column: Column name storing class labels.
            classes: Allowed class names.

        Raises:
            FileNotFoundError: If CSV file does not exist.
            ValueError: If required columns are missing or no valid samples.
        """

        self.csv_file = Path(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.cache_images = cache_images
        self.image_mode = image_mode
        self.path_column = path_column
        self.label_column = label_column
        self.split_column = split_column
        self.split = split

        classes_tuple = tuple(classes)
        if len(classes_tuple) == 0:
            raise ValueError("`classes` cannot be empty.")
        self.classes = list(classes_tuple)
        self.class_to_idx = {label: idx for idx, label in enumerate(classes_tuple)}

        if not self.csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_file}")

        frame = pd.read_csv(self.csv_file)
        self._validate_columns(frame)
        frame = self._filter_split(frame)
        frame = self._filter_classes(frame)

        if frame.empty:
            raise ValueError(
                f"No samples available after filtering split='{split}' and classes={self.classes}."
            )

        self.data = frame.reset_index(drop=True)
        self._cache: Dict[int, Image.Image] = {}

    def _validate_columns(self, frame: pd.DataFrame) -> None:
        """Validate required CSV columns."""

        required = {self.path_column, self.label_column}
        missing = required.difference(frame.columns)
        if missing:
            cols = ", ".join(sorted(missing))
            raise ValueError(f"Missing required CSV columns: {cols}")

    def _filter_split(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Filter dataframe by split if requested."""

        split = self.split.lower()
        if split == "all":
            return frame

        if self.split_column not in frame.columns:
            raise ValueError(
                f"Split filtering requested but '{self.split_column}' column is missing."
            )
        return frame[frame[self.split_column].astype(str).str.lower() == split]

    def _filter_classes(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Keep only rows belonging to configured classes."""

        labels = frame[self.label_column].astype(str)
        valid = labels.isin(self.class_to_idx)
        return frame[valid]

    def __len__(self) -> int:
        """Return number of samples."""

        return len(self.data)

    def _resolve_image_path(self, rel_path: str) -> Path:
        """Resolve an image path from CSV to an absolute path."""

        return self.root_dir / rel_path

    def _load_image(self, idx: int, path: Path) -> Image.Image:
        """Load image with optional in-memory caching."""

        if self.cache_images and idx in self._cache:
            return self._cache[idx].copy()

        with Image.open(path) as image:
            converted = image.convert(self.image_mode)

        if self.cache_images:
            self._cache[idx] = converted.copy()
        return converted

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        """Return `(image, label)` for a single sample.

        Args:
            idx: Dataset index.

        Returns:
            Transformed image tensor (or PIL image when transform is None) and class index.
        """

        row = self.data.iloc[idx]
        rel_path = str(row[self.path_column])
        label_name = str(row[self.label_column])
        image_path = self._resolve_image_path(rel_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = self._load_image(idx, image_path)
        if self.transform is not None:
            image = self.transform(image)

        return image, self.class_to_idx[label_name]


def _resolve_num_workers(num_workers: Optional[int]) -> int:
    """Resolve a safe default worker count."""

    if num_workers is not None:
        return max(0, int(num_workers))

    cpu_count = os.cpu_count() or 1
    # Keep a small reserve for system responsiveness.
    return max(0, min(8, cpu_count - 1))


def create_dataloader(
    dataset: Dataset,
    config: DataLoaderConfig,
    device: Optional[torch.device] = None,
) -> DataLoader:
    """Create an optimized DataLoader with sensible performance defaults.

    Args:
        dataset: Input dataset.
        config: DataLoader configuration.
        device: Current torch device for pin-memory auto selection.

    Returns:
        Configured `torch.utils.data.DataLoader`.
    """

    workers = _resolve_num_workers(config.num_workers)
    use_cuda = bool(device is not None and device.type == "cuda")
    pin_memory = use_cuda if config.pin_memory is None else config.pin_memory

    loader_kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": config.batch_size,
        "shuffle": config.shuffle,
        "num_workers": workers,
        "pin_memory": pin_memory,
        "drop_last": config.drop_last,
    }

    if workers > 0:
        loader_kwargs["persistent_workers"] = config.persistent_workers
        loader_kwargs["prefetch_factor"] = max(1, int(config.prefetch_factor))

    return DataLoader(**loader_kwargs)


def describe_class_distribution(dataset: COVIDCXNetDataset) -> Dict[str, int]:
    """Return class counts for quick dataset sanity checks."""

    counts: Dict[str, int] = {name: 0 for name in dataset.class_to_idx}
    values = dataset.data[dataset.label_column].astype(str).tolist()
    for label in values:
        if label in counts:
            counts[label] += 1
    return counts
