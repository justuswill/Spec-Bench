import torch
import numpy as np
from typing import Any, List, Optional
from lightning.pytorch import LightningDataModule
from datasets import load_dataset
from tqdm import tqdm
import os

CACHE_DIR = 'data_cache'


class MaskCreationError(Exception):
    pass


class ChatDataset(torch.utils.data.Dataset):
    """
    Processes chat datasets like ultrachat where conversations are stored as alternating
    user/assistant messages and formats them using the tokenizer's chat template.
    """

    def __init__(self, dataset_name, split, cache_dir, tokenizer, max_sequence_length: int, add_assistant_prompt: bool = False):
        self.data = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
        self.tokenizer = tokenizer
        self.add_assistant_prompt = add_assistant_prompt

        lookup_cache_file = os.path.join(
            CACHE_DIR,
            dataset_name.replace('/', '_') + f'_{split}_{max_sequence_length}_chat_lookup.npy'
        )

        if os.path.exists(lookup_cache_file):
            print(f"Loading chat lookup from {lookup_cache_file}")
            self.lookup_array = np.load(lookup_cache_file)
        else:
            print(f"Creating chat lookup and saving to {lookup_cache_file}")
            lookup_list = []

            for chat_idx, item in enumerate(tqdm(self.data, desc="Processing chat dataset")):
                try:
                    # Convert conversation data to chat format
                    conversation = self._convert_to_chat_format(item['data'])

                    # Create segments of odd length (ending on user message)
                    for segment_length in range(1, len(conversation) + 1, 2):
                        segment = conversation[:segment_length]
                        if self.add_assistant_prompt:
                            segment.append({"role": "assistant", "content": ""})

                        # Format using tokenizer's chat template
                        formatted_chat = self.tokenizer.apply_chat_template(
                            segment,
                            tokenize=False,
                            add_generation_prompt=False
                        )

                        # Check if it fits within max_sequence_length
                        token_length = len(self.tokenizer(formatted_chat)['input_ids'])

                        if token_length <= max_sequence_length:
                            # Store (chat_index, segment_length) for this valid segment
                            lookup_list.append([chat_idx, segment_length])
                            break

                except Exception as e:
                    # Skip conversations that can't be processed
                    print(f"Skipping conversation {chat_idx} due to error: {e}")
                    continue

            self.lookup_array = np.array(lookup_list)
            np.save(lookup_cache_file, self.lookup_array)
            raise MaskCreationError("Chat dataset lookup file was just created, please rerun to load it.")

    def _convert_to_chat_format(self, conversation_data: Any) -> List[dict]:
        """
        Convert the ultrachat data format (alternating user/assistant messages)
        to the standard chat format expected by tokenizers.

        Args:
            conversation_data: List of strings where even indices are user messages
                             and odd indices are assistant messages

        Returns:
            List of dictionaries with 'role' and 'content' keys
        """
        conversation = []
        for i, message in enumerate(conversation_data):
            if i % 2 == 0:  # Even indices are user messages
                conversation.append({"role": "user", "content": message})
            else:  # Odd indices are assistant messages
                conversation.append({"role": "assistant", "content": message})
        return conversation

    def __len__(self):
        return len(self.lookup_array)

    def __getitem__(self, index) -> str:
        """
        Returns the formatted chat conversation segment as a string.
        """
        chat_idx, segment_length = self.lookup_array[index]
        item = self.data[chat_idx]
        conversation = self._convert_to_chat_format(item['data'])

        # Get the segment of the specified length + assistant prompt
        segment = conversation[:segment_length]
        if self.add_assistant_prompt:
            segment.append({"role": "assistant", "content": ""})

        # Use the tokenizer's chat template to format the conversation segment
        formatted_chat = self.tokenizer.apply_chat_template(
            segment,
            tokenize=False,
            add_generation_prompt=False
        )

        return formatted_chat


class ChatDataModule(LightningDataModule):
    """
    DataLoader wrapper for ChatDataset that processes multi-turn conversations
    using the tokenizer's chat template.
    """

    def __init__(self, dataset_name: str, tokenizer, max_sequence_length: int,
                 train_shuffle: bool = True, splits: List[str] | None = None,
                 chat_template: Optional[str] = None,
                 add_assistant_prompt: bool = False,
                 padding: bool = True, cache_dir: str = CACHE_DIR, **kwargs):
        super().__init__()
        self.dataset_name = dataset_name
        if splits is None:
            splits = ['train', 'valid', 'test']
        self.splits = splits
        self.kwargs = kwargs
        self.tokenizer = tokenizer
        if chat_template is not None:
            self.tokenizer.chat_template = chat_template
        self.padding = padding
        self.max_sequence_length = max_sequence_length
        self.add_assistant_prompt = add_assistant_prompt
        self.train_shuffle = train_shuffle
        self.cache_dir = cache_dir

    def collate_fn(self, batch):
        """
        Tokenize the formatted chat conversations.
        """
        tokenizer_out = self.tokenizer(batch, return_tensors='pt', padding=self.padding)
        return {
            'prompt_ids': tokenizer_out['input_ids'],
            'prompt_mask': tokenizer_out['attention_mask'],
        }

    def setup(self, stage: Any = None) -> None:
        """
        Setup the chat datasets for each split.
        """
        datasets = {}
        any_raised = False

        for split in self.splits:
            try:
                datasets[split] = ChatDataset(
                    self.dataset_name,
                    split=split,
                    cache_dir=self.cache_dir,
                    tokenizer=self.tokenizer,
                    max_sequence_length=self.max_sequence_length,
                    add_assistant_prompt=self.add_assistant_prompt,
                )
            except MaskCreationError as e:
                any_raised = e

        if any_raised is not False:
            raise any_raised

        self.train_dataset = datasets.get("train")
        self.val_dataset = datasets.get("valid")
        self.test_dataset = datasets.get("test")
        
        # If no validation data exists but test data does, use test data for validation
        if self.val_dataset is None and self.test_dataset is not None:
            self.val_dataset = self.test_dataset
            print("No validation data found, using test data for validation")

        # If no validation data exists but test data does, use test data for validation
        if self.val_dataset is None and self.test_dataset is not None:
            self.val_dataset = self.test_dataset
            print("No validation data found, using test data for validation")

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        if self.train_dataset is None:
            raise ValueError("Train dataset not available")
        return torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=self.train_shuffle,
            collate_fn=self.collate_fn,
            **self.kwargs
        )

    def val_dataloader(self) -> Optional[torch.utils.data.DataLoader]:
        if self.val_dataset is None:
            return None
        return torch.utils.data.DataLoader(
            self.val_dataset,
            shuffle=False,
            collate_fn=self.collate_fn,
            **self.kwargs
        )

    def test_dataloader(self) -> Optional[torch.utils.data.DataLoader]:
        if self.test_dataset is None:
            return None
        return torch.utils.data.DataLoader(
            self.test_dataset,
            shuffle=False,
            collate_fn=self.collate_fn,
            **self.kwargs
        )
