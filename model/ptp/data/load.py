import random
from pathlib import Path
import os
import numpy as np
from tqdm import tqdm

import torch
from datasets import Dataset
from lightning import LightningDataModule
from transformers import AutoTokenizer

from ptp.data.sampler import CoordinatedCompletionSampler


class PregeneratedDataset(torch.utils.data.Dataset):
    """
    Assuming the following dataset structure:

    {
        prompt_name: (prompt_seq_len),
        completions_name: (num_completions, completion_seq_len),
        'left_bin_edges': (num_completions, completion_seq_len),
        'right_bin_edges': (num_completions, completion_seq_len),
    }

    This dataset randomly selects one of the completions for each prompt,
    dividing it randomly into a prompt part and a completion part.

    This is where the model completion length is actually enforced.
    It is usually shorter than the completion_seq_len in the dataset,
    which is the full answer by the teacher model.

    :param train_completion_len: int: The completion length fed to the model.
    :param max_sequence_length: int: Maximum total sequence length (prompt + completion)
    :param tokenizer: Tokenizer for length calculation
    """

    def __init__(self, completion_dataset, train_completion_len: int, eos_token_id: int,
                 prompt_name: str = "input", completions_name: str = "completions",
                 max_sequence_length: int = None, tokenizer=None, experiment_dir: str | Path = None):
        super().__init__()
        self.completion_dataset = completion_dataset

        self.prompt_name = prompt_name
        self.completions_name = completions_name

        self.train_completion_len = train_completion_len
        self.eos_token_id = eos_token_id
        self.max_sequence_length = max_sequence_length
        self.tokenizer = tokenizer
        self.experiment_dir = experiment_dir

        # Apply filtering if max_sequence_length is specified
        if max_sequence_length is not None and tokenizer is not None:
            print(f"Applying sequence length filtering with max_sequence_length={max_sequence_length}")
            self._apply_sequence_length_filtering()
        else:
            # No filtering - use all indices
            self.useful_indices = np.arange(len(self.completion_dataset)).astype(int)

    def _apply_sequence_length_filtering(self):
        """Apply filtering based on sequence length, similar to prompt_scheme.py"""
        # Create cache directory - use experiment_dir if available, otherwise default
        if self.experiment_dir is not None:
            cache_dir = Path(self.experiment_dir) / 'data_cache'
        else:
            cache_dir = Path('data_cache')
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Create cache filename based on dataset path and parameters
        dataset_path = str(self.completion_dataset._info.dataset_name) if hasattr(self.completion_dataset, '_info') else 'pregenerated'
        mask_cache_file = cache_dir / f'{dataset_path}_{self.max_sequence_length}_mask.npy'
        # print(f'='*50)
        # print(f"dataset_path: {dataset_path}")
        # print(f"cache_dir: {cache_dir}")
        # print(f"mask_cache_file: {mask_cache_file}")
        # print(f'='*50)
        if mask_cache_file.exists():
            print(f"Loading sequence length mask from {mask_cache_file}")
            boolean_mask = np.load(mask_cache_file)
        else:
            print(f"Creating sequence length mask and saving to {mask_cache_file}")
            boolean_mask = np.array([
                self._check_sequence_length(item) <= self.max_sequence_length
                for item in tqdm(self.completion_dataset, desc="Filtering dataset by sequence length")
            ])
            np.save(mask_cache_file, boolean_mask)
            print("Sequence length mask created. Please rerun to load the filtered dataset.")
            raise ValueError("Dataset mask file was just created, please rerun to load it.")

        self.useful_indices = np.where(boolean_mask)[0].astype(int)
        print(f"Filtered dataset: {len(self.useful_indices)}/{len(self.completion_dataset)} sequences fit within max_sequence_length={self.max_sequence_length}")

    def _check_sequence_length(self, item):
        """Check the total sequence length for a given item"""
        # Get the prompt length
        prompt_length = len(item[self.prompt_name])
        # print(f'prompt_name: {self.prompt_name}')
        # print(f'prompt_length: {len(item[self.prompt_name])}')
        # Get the maximum completion length (worst case scenario)
        max_completion_length = max(
            len(completion) for completion in item[self.completions_name]
        )

        # Total sequence length = prompt + completion
        return prompt_length + max_completion_length

    def __len__(self):
        return len(self.useful_indices)

    def __getitem__(self, selector):
        if isinstance(selector, int):
            # Map the filtered index to the original dataset index
            original_index = int(self.useful_indices[selector])
            #fix this branch bc random completion index can cause issues.
            # if completion_idx is close to len(entry[self.completions_name]) - 1, then can't enforce completion of length 16.
            entry = self.completion_dataset[original_index]
            completions = entry[self.completions_name]
            completion_idx = random.randint(0, len(completions) - 1)
            split_idx = random.randint(0, len(completions[0]) - 1)
        else:
            prompt_idx, completion_idx, cutoff = selector
            # Map the filtered index to the original dataset index
            original_index = int(self.useful_indices[prompt_idx])
            entry = self.completion_dataset[original_index]
            completions = entry[self.completions_name]
            completion_idx = completion_idx % len(completions)
            split_idx = max(0, int(cutoff * (completions.shape[1] - self.train_completion_len)))

        # Do not pad with EOS, because the model predictions might have been cut off in the pregeneration
        completion = completions[completion_idx]
        left_bin_edges = entry["left_bin_edges"][completion_idx]
        right_bin_edges = entry["right_bin_edges"][completion_idx]
        # print(f'='*50)
        # print(f"prompt_ids: {prompt_ids.shape}")
        # print(f'completion ids: {len(completion[split_idx:split_idx + self.train_completion_len])}')
        # print(f'='*50)
        return {
            "prompt_ids": torch.cat([
                entry[self.prompt_name],
                completion[:split_idx]
            ]),
            "prompt_mask": torch.cat([
                torch.ones_like(entry[self.prompt_name], dtype=torch.bool),
                torch.ones(split_idx, dtype=torch.bool)
            ]),
            "completion_ids": completion[split_idx:split_idx + self.train_completion_len],
            "left_bin_edges": left_bin_edges[split_idx:split_idx + self.train_completion_len],
            "right_bin_edges": right_bin_edges[split_idx:split_idx + self.train_completion_len],
        }


class PregeneratedDataModule(LightningDataModule):
    def __init__(self, root_dir: str | Path, train_completion_len: int, tokenizer_id: str,
                 batch_size: int, max_sequence_length: int = None, experiment_dir: str | Path = None,
                 prompt_name: str = "input", completions_name: str = "completions", **kwargs):
        super().__init__()
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)
        self.root_dir = root_dir

        self.train_completion_len = train_completion_len
        self.max_sequence_length = max_sequence_length
        self.experiment_dir = experiment_dir
        if isinstance(tokenizer_id, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        else:
            tokenizer = tokenizer_id
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        self.prompt_name = prompt_name
        self.completions_name = completions_name

        self.datasets = {}
        self.batch_size = batch_size
        self.kwargs = kwargs

    def collate_fn(self, batch):
        """Custom collate function to handle variable length sequences."""
        # Find the maximum length in the batch
        max_prompt_len = max(item["prompt_ids"].shape[0] for item in batch)

        # Pad all sequences to the same length
        padded_batch = {}
        for key in batch[0].keys():
            if key == "prompt_ids":
                # Pad prompt_ids with pad_token_id
                padded_tensors = []
                for item in batch:
                    tensor = item[key]
                    if tensor.shape[0] < max_prompt_len:
                        padding = torch.full((max_prompt_len - tensor.shape[0],), self.pad_token_id, dtype=tensor.dtype)
                        padded_tensor = torch.cat([tensor, padding])
                    else:
                        padded_tensor = tensor
                    padded_tensors.append(padded_tensor)
                padded_batch[key] = torch.stack(padded_tensors)
            elif key == "prompt_mask":
                # Pad prompt_mask with False (no attention to padding tokens)
                padded_tensors = []
                for item in batch:
                    tensor = item[key]
                    if tensor.shape[0] < max_prompt_len:
                        padding = torch.zeros(max_prompt_len - tensor.shape[0], dtype=tensor.dtype)
                        padded_tensor = torch.cat([tensor, padding])
                    else:
                        padded_tensor = tensor
                    padded_tensors.append(padded_tensor)
                padded_batch[key] = torch.stack(padded_tensors)
            else:
                # For completion_ids, left_bin_edges, right_bin_edges - these should already be the same length
                padded_batch[key] = torch.stack([item[key] for item in batch])

        return padded_batch

    def setup(self, stage: str) -> None:
        datasets = {}
        for split in ["train", "val", "test"]:
            if (self.root_dir / split).exists():
                dataset = Dataset.load_from_disk(self.root_dir / split)
                dataset.set_format(type="torch")
                datasets[split] = PregeneratedDataset(
                    dataset,
                    train_completion_len=self.train_completion_len,
                    eos_token_id=self.eos_token_id,
                    prompt_name=self.prompt_name,
                    completions_name=self.completions_name,
                    max_sequence_length=self.max_sequence_length,
                    tokenizer=self.tokenizer,
                    experiment_dir=self.experiment_dir
                )
            elif split == "train":
                raise FileNotFoundError(f"Training dataset not found in {self.root_dir / split}")

        # If no validation data exists but test data does, use test data for validation
        if "val" not in datasets and "test" in datasets:
            datasets["val"] = datasets["test"]
            print("No validation data found, using test data for validation")


        # If no validation data exists but test/train data does, use test/train data for validation
        if not 'val' in datasets.keys() and 'test' in datasets.keys():
            datasets['val'] = datasets['test']
            print("No validation data found, using test data for validation")
        if not 'val' in datasets.keys():
            datasets['val'] = datasets['train']
            print("No validation data found, using train data for validation")
        self.datasets = datasets

    def train_dataloader(self):
        dataset = self.datasets["train"]
        # Filter out parameters that shouldn't be passed to DataLoader
        dataloader_kwargs = {k: v for k, v in self.kwargs.items()
                           if k not in ['max_sequence_length', 'experiment_dir']}
        return torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size,
            sampler=CoordinatedCompletionSampler(len(dataset), self.batch_size, shuffle=True),
            collate_fn=self.collate_fn,
            **dataloader_kwargs
        )

    def val_dataloader(self):
        if "val" in self.datasets:
            dataset = self.datasets["val"]
            # Filter out parameters that shouldn't be passed to DataLoader
            dataloader_kwargs = {k: v for k, v in self.kwargs.items()
                               if k not in ['max_sequence_length', 'experiment_dir']}
            return torch.utils.data.DataLoader(
                dataset, batch_size=self.batch_size,
                sampler=CoordinatedCompletionSampler(len(dataset), self.batch_size, shuffle=False),
                collate_fn=self.collate_fn,
                **dataloader_kwargs
            )
        return []

    def test_dataloader(self):
        if "test" in self.datasets:
            dataset = self.datasets["test"]
            # Filter out parameters that shouldn't be passed to DataLoader
            dataloader_kwargs = {k: v for k, v in self.kwargs.items()
                               if k not in ['max_sequence_length', 'experiment_dir']}
            return torch.utils.data.DataLoader(
                dataset, batch_size=self.batch_size,
                sampler=CoordinatedCompletionSampler(len(dataset), self.batch_size, shuffle=False),
                collate_fn=self.collate_fn,
                **dataloader_kwargs
            )
        return []
