import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PretrainedConfig, GenerationMixin, PreTrainedTokenizer
from transformers.modeling_outputs import CausalLMOutput


class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {
            "prompt_ids": torch.tensor([0]),
            "prompt_mask": torch.tensor([1]),
            "completion_ids": self.data[index],
        }


class IncreasingSequenceDataModule(LightningDataModule):
    def __init__(self, batch_size=32, seq_len=6, vocab_size=4, num_samples=1024):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_samples = num_samples

        self.train_data = None
        self.val_data = None

    def make_data(self, size):
        data = torch.cat([
            torch.randint(0, self.vocab_size, (size, self.seq_len))
        ], dim=1)
        for i in range(self.vocab_size):
            # After the first occurence of token i, all subsequent tokens are at least i
            mask = (data == i).cumsum(dim=1).clamp(0, 1).bool()
            data[mask] = torch.where(
                data[mask] < i,
                i,
                data[mask]
            )
        return data

    def setup(self, stage=None):
        self.train_data = TokenDataset(self.make_data(self.num_samples))
        self.val_data = TokenDataset(self.make_data(self.num_samples // 10))

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)


class IncreasingSequenceTokenizer:
    def __init__(self, eos_token_id):
        self.eos_token_id = eos_token_id
        self.pad_token_id = None

    def decode(self, token_ids, skip_special_tokens=False):
        if not torch.is_tensor(token_ids):
            token_ids = torch.tensor(token_ids)
        return " ".join(
            str(t.item())
            for t in token_ids
            if t.item() != self.eos_token_id or not skip_special_tokens
        )


class IncreasingSequenceConfig(PretrainedConfig):
    model_type = "increasing-sequence"
    def __init__(self, vocab_size=100, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size


class IncreasingSequenceTeacherModel(PreTrainedModel, GenerationMixin):
    config_class = IncreasingSequenceConfig

    def __init__(self, config):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.tokenizer = IncreasingSequenceTokenizer(-1)

    def forward(self, input_ids, labels=None, attention_mask=None, return_dict=True):
        assert attention_mask is None or (attention_mask == 1).all()

        batch_size, seq_len = input_ids.shape
        probs = torch.full((batch_size, seq_len, self.vocab_size), 1e-5, device=input_ids.device)
        for i in range(batch_size):
            for j in range(seq_len):
                current_token = input_ids[i, j].item()
                probs[i, j, current_token:] = 1.0  # logit for same or larger token
        probs /= probs.sum(dim=-1, keepdim=True)  # normalize to get probabilities
        logits = probs.log()

        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}

    @property
    def device(self):
        return torch.device("cpu")
