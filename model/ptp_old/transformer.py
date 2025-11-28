import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.checkpoint import checkpoint
import torch.nn as nn


class CustomCheckpointWrapper(nn.Module):
    def __init__(self, layer, use_reentrant: bool = False, preserve_rng_state: bool = True):
        super().__init__()
        self.layer = layer
        self.use_reentrant = use_reentrant
        self.preserve_rng_state = preserve_rng_state

    def _call(self, *args, **kwargs):
        return self.layer(*args, **kwargs)

    def forward(self, *args, **kwargs):
        # kwargs are supported in modern PyTorch. If your env is old, wrap inputs into a tuple-only signature.
        return checkpoint(
            self._call,
            *args,
            use_reentrant=self.use_reentrant,
            preserve_rng_state=self.preserve_rng_state,
            **kwargs,
        )

    def __getattr__(self, name):
        """Forward attribute access to the wrapped layer"""
        if name in ['layer', 'use_reentrant', 'preserve_rng_state']:
            return super().__getattr__(name)
        return getattr(self.layer, name)


# Example: wrap every 3rd layer in model.model.layers
def enable_custom_checkpointing(model, every_n: int = 3, start_index: int = 0,
                                use_reentrant: bool = False, preserve_rng_state: bool = True, verbose: bool = True):
    """
    Wraps every Nth block under model.model.layers with a checkpoint wrapper.
    Adjust the path if your model stores blocks elsewhere (e.g., model.transformer.h for GPT-2).
    Returns number of layers wrapped.
    """
    # Turn off use_cache during training when using checkpointing
    if hasattr(model, "config") and getattr(model.config, "use_cache", None) is True:
        model.config.use_cache = False

    # You may need to change this path depending on the HF architecture you use.
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise AttributeError(
            "Could not find 'model.model.layers'. Adjust enable_custom_checkpointing() for your model arch.")

    layers = model.model.layers
    wrapped = 0
    for i, layer in enumerate(layers):
        if i >= start_index and every_n > 0 and ((i - start_index) % every_n == 0):
            if verbose:
                print(f"[GC] Wrapping layer index {i} with checkpoint wrapper")
            layers[i] = CustomCheckpointWrapper(layer,
                                                use_reentrant=use_reentrant,
                                                preserve_rng_state=preserve_rng_state)
            wrapped += 1
    return wrapped


class TransformerModel(torch.nn.Module):
    def __init__(self, model_id, reset_parameters=False, dtype: torch.dtype | str = torch.float32,
                 use_gradient_checkpointing: bool = False, lora_config=None, **kwargs):
        super().__init__()
        # Convert string dtype to torch dtype
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)

        if isinstance(model_id, torch.nn.Module):
            self.tokenizer = None
            self.model = model_id
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            # Use torch_dtype parameter for Hugging Face compatibility
            self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, **kwargs)
            if reset_parameters:
                self.model = AutoModelForCausalLM.from_config(self.model.config)

        # Enable gradient checkpointing to save memory (trades compute for memory)
        if use_gradient_checkpointing:
            print("Enabling gradient checkpointing")
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        if lora_config is not None:
            from peft import get_peft_model, LoraConfig, TaskType
            print("Applying LoRA adapters")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                **lora_config
            )
            self.model = get_peft_model(self.model, peft_config)

        # print how many layers there are in the model
        # print(f"Number of layers in the model: {len(self.model.model.layers)}")
        # if use_gradient_checkpointing:
        #     wrapped = enable_custom_checkpointing(
        #         self.model,
        #         every_n=1, #gradient_checkpointing_freq,
        #         start_index= 8, #gc_start_index,
        #         use_reentrant=False,
        #         preserve_rng_state= True, #gc_preserve_rng_state,
        #         verbose=True,
        #     )
        #     if wrapped == 0:
        #         print("[GC] Warning: No layers were wrapped. Check the layer path or frequency.")

        self.model.train()

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class BinaryFloatEmbedding(torch.nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.u_adapter = torch.nn.Linear(32, embedding_dim)

    def forward(self, u):
        res = u.to(torch.float32).view(torch.int32)
        bit_masks = 2 ** torch.arange(31, -1, -1, device=u.device, dtype=torch.int32)
        bits_tensor = ((res[..., None] & bit_masks[None, None, :]) != 0).int()
        assert torch.all(
            u == (bits_tensor.to(torch.int32) * bit_masks[None, None, :]).sum(dim=2).to(torch.int32).view(torch.float32)
        )
        return self.u_adapter(bits_tensor.to(torch.float32))


class SawtoothFloatEmbedding(torch.nn.Module):
    def __init__(self, embedding_dim, num_bins=32):
        super().__init__()
        self.bin_widths = 1 / torch.arange(1, num_bins + 1).float()
        self.embedding = torch.nn.Linear(num_bins, embedding_dim)

    def forward(self, u):
        # u is (batch_size, seq_len)
        u = torch.clamp(u, 0.0, 1.0 - 1e-6)
        u = u.unsqueeze(-1)  # (batch_size, seq_len, 1)
        # (batch_size, seq_len, num_bins)
        bin_positions = u % self.bin_widths.to(u.device)[None, None, :]
        return self.embedding(bin_positions)


class QuarterCosEmbedding(torch.nn.Module):
    def __init__(self, embedding_dim, num_frequencies=32):
        super().__init__()
        self.frequencies = torch.arange(1, num_frequencies + 1).float()
        assert self.frequencies.shape == (num_frequencies,), f"{self.frequencies.shape} != {(num_frequencies,)}"
        self.embedding = torch.nn.Linear(num_frequencies, embedding_dim)

    def forward(self, u):
        # u is (batch_size, seq_len)
        u = torch.clamp(u, 0.0, 1.0)
        u = u.unsqueeze(-1)  # (batch_size, seq_len, 1)
        # (batch_size, seq_len, num_frequencies)
        cos_features = torch.cos(self.frequencies.to(u.device)[None, None, :] * torch.pi * u)
        return self.embedding(cos_features)


class MixedTransformerModel(TransformerModel):
    def __init__(self, shift_positions: bool = False, adapter_name="binary", *args, **kwargs):
        super().__init__(*args, **kwargs)
        if adapter_name == "sawtooth":
            adapter_class = SawtoothFloatEmbedding
        elif adapter_name == "quarter_cos":
            adapter_class = QuarterCosEmbedding
        elif adapter_name == "binary":
            adapter_class = BinaryFloatEmbedding
        else:
            raise ValueError(f"Unknown adapter_name {adapter_name}")
        self.u_adapter = adapter_class(self.model.config.hidden_size)
        self.shift_positions = shift_positions

    def forward(self, input_ids: torch.LongTensor, auxiliaries: torch.FloatTensor, attention_mask=None, **kwargs):
        # input_embeds = self.model.model.embed_tokens(input_ids)
        input_embeds = self.model.get_input_embeddings()(input_ids)
        auxiliary_embeds = self.u_adapter(auxiliaries)

        all_embeds = torch.cat([
            input_embeds,
            auxiliary_embeds
        ], dim=1)
        if self.shift_positions:
            assert 'position_ids' not in kwargs, "Cannot use both shift_positions and custom position_ids"
            if attention_mask is None:
                attention_mask = torch.ones(
                    all_embeds.shape[0], all_embeds.shape[1],
                    device=input_ids.device, dtype=torch.bool
                )
            position_ids = attention_mask.cumsum(dim=1) - 1
            # First auxiliary prediction overlaps with last input token
            position_ids[:, input_embeds.shape[-2]:] -= 1
            kwargs['position_ids'] = position_ids
        # return self.model(inputs_embeds=all_embeds, attention_mask=attention_mask, **kwargs)
        # return self.model(inputs_embeds=all_embeds, attention_mask=attention_mask, **kwargs)
        return self.model(inputs_embeds=all_embeds, **kwargs)