import torch
import math
import numpy as np
from typing import Any, List, Optional
from lightning.pytorch import LightningDataModule
from datasets import load_dataset
from tqdm import tqdm
import os

CACHE_DIR = 'data_cache'


class WikiDataset(torch.utils.data.Dataset):
    """
    Processes datasets like wikitext that can be fully loaded in memory,
    obtaining training sequences with random starting positions.
    """

    def __init__(self, dataset_name, dataset_config, split, cache_dir, tokenizer, max_sequence_length: int):
        self.data = load_dataset(dataset_name, dataset_config, split=split, cache_dir=cache_dir)
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.attn = torch.ones(self.max_sequence_length, dtype=int)

        token_cache_file = os.path.join(
            CACHE_DIR,
            dataset_name.replace('/', '_') + f'_{split}_tokenized.npy'
        )

        if os.path.exists(token_cache_file):
            print(f"Loading wiki tokens from {token_cache_file}")
            self.token_array = np.load(token_cache_file)
        else:
            print(f"Creating wiki tokens and saving to {token_cache_file}")
            token_list = []
            # full_text = ''

            for article in tqdm(self.data, desc="Processing wiki dataset"):
                text = '%s\n\n%s\n\n' % (article['title'], article['text'])
                token_list += [np.array(self.tokenizer(text)['input_ids'][1:], dtype=np.uint16)]
                # full_text += text

            self.token_array = np.concatenate(token_list)
            # self.token_array = self.tokenizer(full_text)['input_ids']
            print('Full dataset contains %d tokens' % self.token_array.size)
            np.save(token_cache_file, self.token_array)

        # Use permutation i -> (c * i) mod n,
        # should cover well early, if c / n is far away from a simple fraction
        self.perm_c = math.floor(0.6180 * len(self))
        while math.gcd(self.perm_c, len(self)) != 1:
            self.perm_c += 1

    def __len__(self):
        return len(self.token_array) - self.max_sequence_length + 1

    def __getitem__(self, idx) -> np.array:
        pidx = (self.perm_c * idx) % len(self)
        return {
            'prompt_ids': torch.tensor(self.token_array[pidx:pidx + self.max_sequence_length], dtype=torch.int64),
            'prompt_mask': self.attn
        }


class WikiDataModule(LightningDataModule):
    """
    DataLoader wrapper for WikiDataset.
    """\

    def __init__(self, dataset_name: str, tokenizer, max_sequence_length: int, dataset_config: str = None,
                 train_shuffle: bool = True, splits: List[str] | None = None,
                 padding: bool = True, cache_dir: str = CACHE_DIR, **kwargs):
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        if splits is None:
            splits = ['train', 'valid', 'test']
        self.splits = splits
        self.tokenizer = tokenizer
        self.kwargs = kwargs
        self.padding = padding
        self.max_sequence_length = max_sequence_length
        self.train_shuffle = train_shuffle
        self.cache_dir = cache_dir

    def setup(self, stage: Any = None) -> None:
        """
        Setup the chat datasets for each split.
        """
        datasets = {}
        any_raised = False

        for split in self.splits:
            datasets[split] = WikiDataset(
                self.dataset_name,
                self.dataset_config,
                split=split,
                cache_dir=self.cache_dir,
                tokenizer=self.tokenizer,
                max_sequence_length=self.max_sequence_length,
            )

        self.train_dataset = datasets.get("train")
        self.val_dataset = datasets.get("valid")
        self.test_dataset = datasets.get("test")

        # If no validation data exists but test/train data does, use test/train data for validation
        if self.val_dataset is None and self.test_dataset is not None:
            self.val_dataset = self.test_dataset
            print("No validation data found, using test data for validation")
        if self.val_dataset is None and self.train_dataset is not None:
            self.val_dataset = self.train_dataset
            print("No validation data found, using test data for validation")

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        if self.train_dataset is None:
            raise ValueError("Train dataset not available")
        return torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=self.train_shuffle,
            **self.kwargs
        )

    def val_dataloader(self) -> Optional[torch.utils.data.DataLoader]:
        if self.val_dataset is None:
            return None
        return torch.utils.data.DataLoader(
            self.val_dataset,
            shuffle=False,
            **self.kwargs
        )

    def test_dataloader(self) -> Optional[torch.utils.data.DataLoader]:
        if self.test_dataset is None:
            return None
        return torch.utils.data.DataLoader(
            self.test_dataset,
            shuffle=False,
            **self.kwargs
        )
