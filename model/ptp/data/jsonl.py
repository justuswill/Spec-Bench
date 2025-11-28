from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


class JsonlChatPromptDataset(Dataset):
    def __init__(self, jsonl_path: str | Path, tokenizer_name: str, chat_template: str = None):
        self.jsonl_path = Path(jsonl_path)
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {self.jsonl_path}")

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(tokenizer_name)
        if chat_template is not None:
            self.tokenizer.chat_template = chat_template
        if not hasattr(self.tokenizer, "apply_chat_template"):
            raise AttributeError(
                "Tokenizer must implement `apply_chat_template` to format chat prompts."
            )

        self._records = self._load_records()

    def _load_records(self):
        records: List[List[int]] = []
        with self.jsonl_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON content on line {line_number} of {self.jsonl_path}: {exc}"
                    ) from exc
        return records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> dict:
        return self._records[index]
