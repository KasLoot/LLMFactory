"""PyTorch dataset for manifest-described packed token binaries."""

from __future__ import annotations

import bisect
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class PackedTokenDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        split: str = "train",
        context_length: int = 2048,
        shuffle_files: bool = False,
        seed: int = 42,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.manifest = self._read_manifest(self.manifest_path)
        self.root = self.manifest_path.parent
        self.split = split
        self.context_length = context_length

        if context_length <= 0:
            raise ValueError("context_length must be positive")
        if split not in self.manifest["splits"]:
            raise ValueError(f"Split {split!r} not found in manifest")

        self.dtype = np.dtype(self.manifest["dtype"])
        shard_meta = list(self.manifest["splits"][split]["shards"])
        if shuffle_files:
            random.Random(seed).shuffle(shard_meta)

        self.shards: list[np.memmap] = []
        self.shard_tokens: list[int] = []
        self.block_counts: list[int] = []
        cumulative = 0
        self.cumulative_blocks: list[int] = []

        for meta in shard_meta:
            path = self.root / meta["filename"]
            if not path.is_file():
                raise FileNotFoundError(f"Missing token shard: {path}")
            token_count = int(meta["tokens"])
            mmap = np.memmap(path, dtype=self.dtype, mode="r", shape=(token_count,))
            blocks = max(0, (token_count - 1) // context_length)
            if blocks == 0:
                continue

            self.shards.append(mmap)
            self.shard_tokens.append(token_count)
            self.block_counts.append(blocks)
            cumulative += blocks
            self.cumulative_blocks.append(cumulative)

    @staticmethod
    def _read_manifest(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        if manifest.get("format") != "raw_token_binary_v1":
            raise ValueError(f"Unsupported manifest format in {path}")
        return manifest

    def __len__(self) -> int:
        return self.cumulative_blocks[-1] if self.cumulative_blocks else 0

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        length = len(self)
        if idx < 0:
            idx += length
        if idx < 0 or idx >= length:
            raise IndexError(idx)

        shard_idx = bisect.bisect_right(self.cumulative_blocks, idx)
        prev_blocks = 0 if shard_idx == 0 else self.cumulative_blocks[shard_idx - 1]
        block_idx = idx - prev_blocks
        offset = block_idx * self.context_length

        raw = self.shards[shard_idx][offset : offset + self.context_length + 1]
        input_ids = torch.from_numpy(np.asarray(raw[:-1], dtype=np.int64))
        labels = torch.from_numpy(np.asarray(raw[1:], dtype=np.int64))
        return {"input_ids": input_ids, "labels": labels}
