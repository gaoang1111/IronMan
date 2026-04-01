"""Data loading utilities."""
from __future__ import annotations

import json
from torch.utils.data import Dataset


class JsonlDataset(Dataset):
    """Simple JSONL dataset that reads {"text": "..."} lines."""

    def __init__(self, file_path: str):
        self.texts: list[str] = []
        print(f"Loading data from {file_path}...")
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                self.texts.append(item["text"])
        print(f"Loaded {len(self.texts)} samples.")

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[str, int]:
        return self.texts[idx], int(idx)
