"""
dataset.py — PyTorch Dataset クラス

ImageFolder 形式のディレクトリ（cover/ / content/）から画像を読み込む
ラッパー Dataset を提供する。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset

# ラベル定義（アルファベット順で固定）
CLASS_NAMES = ["content", "cover"]   # 0: content, 1: cover
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}


class IIIFDataset(Dataset):
    """
    ImageFolder 形式のディレクトリから画像を読み込む Dataset。

    ディレクトリ構造:
        root/
          cover/   ← ラベル "cover"
          content/ ← ラベル "content"

    Args:
        root: データセットのルートディレクトリ
        transform: torchvision の Transform。未指定なら ToTensor のみ。
    """

    def __init__(
        self,
        root: str | Path,
        transform: Optional[Callable] = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.samples: list[Tuple[Path, int]] = []

        for class_name, class_idx in CLASS_TO_IDX.items():
            class_dir = self.root / class_name
            if not class_dir.exists():
                continue
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
                for img_path in sorted(class_dir.glob(ext)):
                    self.samples.append((img_path, class_idx))

        if not self.samples:
            raise RuntimeError(
                f"No images found in {self.root}. "
                "Expected subdirectories: cover/, content/"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    @property
    def targets(self) -> list[int]:
        """WeightedRandomSampler 用にラベルリストを返す。"""
        return [label for _, label in self.samples]

    def class_distribution(self) -> dict[str, int]:
        """各クラスのサンプル数を返す。"""
        dist: dict[str, int] = {c: 0 for c in CLASS_NAMES}
        for _, label in self.samples:
            dist[CLASS_NAMES[label]] += 1
        return dist
