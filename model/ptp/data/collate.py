import torch


def collate_fn(batch):
    """
    Batch is list of dicts:
    {
        "prompt_ids": (inconsistent_prompt_seq_len,),
        "completion_ids": (consistent_prompt_completion_len,),
        "left_bin_edges": (consistent_prompt_completion_len,),
        "right_bin_edges": (consistent_prompt_completion_len,)
    }

    :param batch:
    :return:
        A single dict, where the prompt_ids are padded with 0s on the right, the completions do not need to be padded.
        A prompt_mask indicates which entries in the prompt are meaningful.
    """
    # Extract prompt_ids and find max length for padding
    prompt_ids_list = [item["prompt_ids"] for item in batch]
    max_prompt_len = max(len(prompt_ids) for prompt_ids in prompt_ids_list)

    # Pad prompt_ids with 0s on the right and create mask
    padded_prompt_ids = []
    prompt_masks = []

    for prompt_ids in prompt_ids_list:
        prompt_len = len(prompt_ids)
        # Create mask (1 for real tokens, 0 for padding)
        mask = torch.ones(prompt_len, dtype=torch.bool)
        if prompt_len < max_prompt_len:
            # Pad with zeros
            padding = torch.zeros(max_prompt_len - prompt_len, dtype=prompt_ids.dtype)
            padded_prompt = torch.cat([prompt_ids, padding])
            # Extend mask with False for padding
            mask_padding = torch.zeros(max_prompt_len - prompt_len, dtype=torch.bool)
            mask = torch.cat([mask, mask_padding])
        else:
            padded_prompt = prompt_ids

        padded_prompt_ids.append(padded_prompt)
        prompt_masks.append(mask)

    # Stack all tensors
    result = {
        "prompt_ids": torch.stack(padded_prompt_ids),
        "prompt_mask": torch.stack(prompt_masks),
        "completion_ids": torch.stack([item["completion_ids"] for item in batch]),
        "left_bin_edges": torch.stack([item["left_bin_edges"] for item in batch]),
        "right_bin_edges": torch.stack([item["right_bin_edges"] for item in batch])
    }

    return result
