import time

from lightning.pytorch import LightningModule
from typing import Literal, List, Mapping, Any
from model.ptp.transformer import TransformerModel, MixedTransformerModel, GatedLinearLora
import random
import torch
import numpy as np
from transformers.cache_utils import DynamicCache, StaticCache


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ParallelSamplingLightningModule(LightningModule):
    def __init__(self, optim_cfg: dict = None, student_cfg: Mapping[str, Any] | None = None, teacher_cfg: Mapping[str, Any] | None = None,
                 student: MixedTransformerModel | None = None, teacher: TransformerModel | None = None,
                 total_max_len: int | None = None, max_completion_length: int | None = None,
                 num_completions_per_prompt: int | None = None,
                 temperature: float | None = None, top_p: float | None = None, top_k: int | None = None,
                 loss_type: Literal['bc', 'kl_rev', 'baseline', 'kl_self', 'uid', 'mtp', 'c-ptp'] = 'bc',
                 pbar_metrics: List[str] | None = None,
                 error_correction = False, student_calls_per_step = 1, tokens_per_student_call = None, tokens_to_fill = None):
        if pbar_metrics is None:
            pbar_metrics = ['loss', 'accuracy', 'correct']
        super().__init__()
        if (student is None) == (student_cfg is None):
            raise ValueError("Exactly one of student and student_cfg must be provided")
        self.student_cfg = student_cfg
        self.student: MixedTransformerModel | None = student
        self.teacher_cfg = teacher_cfg
        self.teacher: TransformerModel | None = teacher

        self.optim_cfg = optim_cfg
        self.loss_type = loss_type

        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

        self.num_completions_per_prompt = num_completions_per_prompt
        self.total_max_len = total_max_len
        self.max_completion_length = max_completion_length

        self.error_correction = error_correction
        self.student_calls_per_step = student_calls_per_step
        self.tokens_per_student_call = tokens_per_student_call
        self.tokens_to_fill = tokens_to_fill
        assert not error_correction or (tokens_to_fill is not None and tokens_per_student_call is not None),\
            'must specify number of tokens to predict per student call and total tokens to fill iteratively'

        self.pbar_metrics = pbar_metrics

        # Optimized inference
        self.past_kv_cache = None
        self.ones_tpsc = torch.ones([1, self.tokens_per_student_call], dtype=int, device=device)
        self.arange_tpscp1 = torch.arange(self.tokens_per_student_call + 1)
        self.arange_21 = torch.arange(21)
        self.inf = torch.tensor(float('inf'))

        Y = np.load('data/Y.npy')
        YY = Y.argmin(axis=1)
        YY[Y.all(axis=1)] = 20
        self.hist_base = torch.tensor(np.unique(YY, return_counts=True)[1] / YY.shape[0])
    
    def configure_model(self) -> None:
        if self.student is None:
            self.student = MixedTransformerModel(**self.student_cfg)
        if self.teacher is None and self.teacher_cfg is not None:
            self.teacher = TransformerModel(**self.teacher_cfg).eval()

    def configure_optimizers(self):
        config = self.optim_cfg
        optimizer = {}
        active_parameters = [p for p in self.student.parameters() if p.requires_grad]
        optimizer["optimizer"] = torch.optim.AdamW(
            active_parameters,
            lr=config["lr"],
        )
        if config.get("lr_warmup", 0) > 0:
            warmup_steps = config["lr_warmup"]
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps  # Linear warmup
                else:
                    return 1.0  # keep lr constant after warmup

            optimizer["lr_scheduler"] = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer["optimizer"], lr_lambda),
                "interval": "step",
                "frequency": 1,
            }
        return optimizer

    def _log_metrics(self, prefix, metrics, sync_dist=False):
        pbar_metrics = {}
        plot_metrics = {}
        other_metrics = {}
        for key, v in metrics.items():
            prefixed_key = f"{prefix}/{key}"
            if key in self.pbar_metrics:
                pbar_metrics[prefixed_key] = v
            elif (isinstance(v, float) or isinstance(v, int)) or v.shape == torch.Size([]):
                other_metrics[prefixed_key] = v
            else:
                # for idx, vi in enumerate(v):
                #    other_metrics[f"{prefixed_k}_{idx}"] = vi
                # plot_metrics[prefixed_key] = v
                pass
        self.log_dict(pbar_metrics, prog_bar=True, sync_dist=sync_dist)
        for k, v in plot_metrics.items():
            if v.ndim == 1 and self.trainer.logger is not None:
                # Convert to CPU list to prevent GPU memory from being held by wandb
                v_cpu = v.detach().cpu().tolist()
                self.trainer.logger.log_table(key=k, data=list(enumerate(v_cpu)), columns=['position', k])
        self.log_dict(other_metrics, prog_bar=False, sync_dist=sync_dist)

    def training_step(self, batch, batch_idx=None):
        metrics = self.forward(batch, batch_idx)

        # Extract loss (keep gradients for backprop)
        loss = metrics['loss']

        # Detach all metrics for logging to prevent memory leak
        metrics_logged = {k: v.detach() if isinstance(v, torch.Tensor) else v
                         for k, v in metrics.items()}
        metrics_logged['lr'] = self.optimizers().param_groups[0]['lr']
        self._log_metrics('train', metrics_logged)

        # Periodically clear CUDA cache to combat fragmentation
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()

        # Return only loss for backprop (Lightning's requirement)
        return loss

    def validation_step(self, batch, batch_idx=None):
        if self.error_correction:
            metrics = self.timed_error_correction(batch, self.tokens_to_fill)
        else:
            metrics = self.forward(batch, batch_idx, eval=True)
        # Detach all validation metrics (no gradients needed)
        metrics_logged = {k: v.detach() if isinstance(v, torch.Tensor) else v
                         for k, v in metrics.items()}
        self._log_metrics('val', metrics_logged, sync_dist=True)

    def batch_from_teacher_completions(self, batch):
        assert batch['input_ids'].shape[0] == 1, "Batch size must be 1 for now"
    
        # This is (batch_size, seq_len,) shape
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        # Always maximize out the sequence length
        completion_length = min(self.total_max_len - input_ids.shape[-1], self.max_completion_length)
        assert completion_length > 0, f"Input too long: {input_ids.shape[-1]} >= {self.total_max_len}"

        # Get completion from teacher
        with torch.no_grad():
            self.teacher.eval()
            completion = self.teacher.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_scores=True, return_dict_in_generate=True,
                temperature=self.temperature, top_p=self.top_p, top_k=self.top_k,
                do_sample=True, max_new_tokens=completion_length,
                num_return_sequences=self.num_completions_per_prompt,
            )
        # Always predict rest of sequence, even if early stopped some sequences
        attention_mask = torch.cat([
            attention_mask.repeat_interleave(self.num_completions_per_prompt, dim=0),
            torch.ones(
                (attention_mask.shape[0] * self.num_completions_per_prompt, completion_length),
                device=attention_mask.device, dtype=attention_mask.dtype
            )
        ], dim=1)
        # Could differ if all sequences terminated early
        completion_length = completion.sequences.shape[1] - input_ids.shape[1]
        random_offset = random.randint(0, completion_length - 1)
        completion_length -= random_offset
        # Input = prompt + completion until break
        student_input_ids = completion.sequences[:, :-completion_length]
        del input_ids
        # Each token is associated with its logits (used for sampling)
        # (batch_size, completion_length)
        student_tgt_ids = completion.sequences[:, -completion_length:]
        # (batch_size, completion_length, vocab_size)
        tgt_logits = torch.stack(completion.scores[-completion_length:], dim=1)
        assert tgt_logits.shape[1] == completion_length
        assert tgt_logits.shape[0] == student_tgt_ids.shape[0]
        assert tgt_logits.shape[2] == self.student.tokenizer.vocab_size

        # Extract u -> token function
        probs = torch.nn.functional.softmax(tgt_logits, dim=-1)
        cum_probs = probs.cumsum(-1)
        device = self.device
        selector = (torch.arange(student_tgt_ids.shape[0], device=device)[:, None], torch.arange(student_tgt_ids.shape[1], device=device)[None], student_tgt_ids)
        left_bin_edge = cum_probs[selector] - probs[selector]
        right_bin_edge = cum_probs[selector]

        return {
            "input_ids": student_input_ids,
            "attention_mask": attention_mask,
            "completions": student_tgt_ids,
            "left_bin_edges": left_bin_edge,
            "right_bin_edges": right_bin_edge,
        }

    def adapt_logits(self, logits):
        if self.temperature is not None and self.temperature != 1.0:
            logits = logits / self.temperature
        if self.top_k is not None and self.top_k > 0:
            top_k_logits, top_k_indices = torch.topk(logits, k=self.top_k)
            mask = torch.full_like(logits, float('-inf'))
            mask.scatter_(-1, top_k_indices, top_k_logits)
            logits = mask
        if self.top_p is not None and self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > self.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
        return logits

    def adapt_p(self, p):
        top_k_probs, top_k_indices = torch.topk(p, k=self.top_k, dim=-1)
        # remove additional tokens if top_p is more restrictive
        remove = (top_k_probs.cumsum(dim=-1) - top_k_probs) > self.top_p
        top_k_probs = top_k_probs.masked_fill(remove, 0.0)
        # renormalize and scatter back to vocab size
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        p.zero_()
        p.scatter_(-1, top_k_indices, top_k_probs)
        return p

    def batch_from_teacher_scores(self, batch):
        """
        Full text is given, get scores (adapted logits) from teacher.

        :param batch:
        :return:
        """
        # Might not need to ask the teacher if we know everything
        required_entries = {"prompt_ids", "completion_ids", "left_bin_edges", "right_bin_edges"}
        if self.loss_type == "kl":
            required_entries.add("logits")
        missing_entries = required_entries - set(batch.keys())
        if len(missing_entries) == 0:
            return batch
        if self.teacher is None:
            raise ValueError(
                f"Teacher cannot be None if dataset does not contain required inputs. Missing: {missing_entries}"
            )

        # batch_size, seq_len
        prompt_ids = batch['prompt_ids']
        assert prompt_ids.shape[-1] >= 1, "Need to have at least one prompt token"
        prompt_mask = batch['prompt_mask']
        completion_ids = batch['completion_ids']

        with torch.no_grad():
            # Cut off last token because we don't predict any token here
            input_ids = torch.cat([
                prompt_ids,
                completion_ids[:, :-1],
            ], dim=-1)
            if prompt_mask is not None:
                input_mask = torch.cat([
                    prompt_mask,
                    # Completion is always active (actively predict sequences of EOS)
                    torch.ones((completion_ids.shape[0], completion_ids.shape[1]), device=prompt_mask.device, dtype=prompt_mask.dtype),
                ], dim=-1)

            # Ask teacher about all distributions
            self.teacher.eval()
            outputs = self.teacher(
                input_ids=input_ids,
                attention_mask=input_mask,
                return_dict=True,
            )

            # batch_size, completion_seq_len, vocab_size
            logits = outputs.logits[..., prompt_ids.shape[-1] - 1:, :]
            assert logits.shape[-2] == completion_ids.shape[-1], f"{logits.shape} does not match {completion_ids.shape}"
            logits = self.adapt_logits(logits)
            assert logits.dtype == torch.float32, logits.dtype
            probs = torch.softmax(logits, dim=-1)
            selector = (
                torch.arange(probs.shape[0], device=probs.device)[:, None],
                torch.arange(probs.shape[1], device=probs.device)[None, :],
                completion_ids
            )
            cum_probs = probs.cumsum(-1)
            left_bin_edge = cum_probs[selector] - probs[selector]
            right_bin_edge = cum_probs[selector]

            new_batch = {
                "prompt_ids": prompt_ids,
                "prompt_mask": prompt_mask,
                "completion_ids": completion_ids,
                # These are the logits/bin edges for predicting the completion tokens
                "logits": logits,
                "left_bin_edges": left_bin_edge,
                "right_bin_edges": right_bin_edge,
            }
            assert (
                    new_batch['logits'].shape[:2]
                    == new_batch['right_bin_edges'].shape
                    == new_batch['left_bin_edges'].shape
                    == new_batch['completion_ids'].shape
            )
            return new_batch

    def forward(self, batch, batch_idx=None, eval=False):
        """
        in |   out
        ---+----------
        t0 |   p1
        t1 |   p2
        t2 |   p3
        t3 |  X / p4
        u4 | o4 / p5
        u5 | o5 / p6
        u6 | o6 / X
        """
        batch = self.batch_from_teacher_scores(batch)
        prompt_ids = batch['prompt_ids']
        prompt_mask = batch['prompt_mask']
        completion_ids = batch['completion_ids']
        left_bin_edges = batch['left_bin_edges']
        right_bin_edges = batch['right_bin_edges']

        device = prompt_ids.device
        z_shape = completion_ids.shape
        if self.loss_type == 'mtp':
            z_rnd = torch.zeros(z_shape, device=device, dtype=torch.float32)
        elif not eval:
            beta_concentration = torch.tensor(0.3, device=device, dtype=torch.float32)
            z_rnd = torch.distributions.Beta(beta_concentration, beta_concentration).sample(z_shape)
        else:
            z_rnd = torch.rand(z_shape, device=device, dtype=torch.float32)
        auxiliaries = left_bin_edges + (right_bin_edges - left_bin_edges) * z_rnd

        input_mask = torch.cat([
            prompt_mask,
            torch.ones(z_shape, device=prompt_mask.device, dtype=prompt_mask.dtype),
        ], dim=1)
        full_logits = self.student(
            input_ids=prompt_ids,
            attention_mask=input_mask,
            auxiliaries=auxiliaries
        ).logits

        if self.loss_type.startswith('kl') or self.loss_type == 'mtp':
            ###
            ### C-PTP
            ### Train student logits to be identical to teacher
            ###
            # Completion tokens start one before the prompt/completion divide, since we predict the next token
            student_logits = torch.cat([
                full_logits[torch.arange(full_logits.shape[0]), prompt_mask.sum(dim=-1) - 1][:, None],
                full_logits[torch.arange(full_logits.shape[0]), prompt_mask.sum(dim=-1) - 1][:, None],
                full_logits[:, prompt_ids.shape[1]:-1],
            ], dim=1)
            with torch.no_grad():
                # Shift by 1 because we predict the next token
                # u_i -> t_{i+1}
                student_right_bin_edges = torch.softmax(student_logits, dim=-1).cumsum(dim=-1)
                student_predicted = (student_right_bin_edges > auxiliaries[..., None]).max(dim=-1).indices

            if self.loss_type == 'mtp':
                loss = torch.nn.functional.cross_entropy(
                    student_logits.transpose(1, 2),
                    completion_ids,
                )
            else:
                tgt_logits = batch['logits']
                loss = torch.nn.functional.kl_div(
                    torch.nn.functional.log_softmax(student_logits, dim=-1),
                    torch.nn.functional.log_softmax(tgt_logits, dim=-1),
                    reduction='batchmean', log_target=True
                )
        elif self.loss_type.startswith('bc'):
            ###
            ### O-PTP
            ### Train student to incorporate the sampling process
            ###
            # Directly predict t_i at output i (shifted compared to usual language modeling)
            student_logits = full_logits[:, prompt_ids.shape[1]:]
            if self.loss_type == 'bc':
                # One-hot cross-entropy loss
                loss = torch.nn.functional.cross_entropy(
                    student_logits.transpose(1, 2),
                    completion_ids,
                )
            elif self.loss_type == 'bce':
                # Binary cross-entropy loss on the sampled token
                one_hot_x = torch.zeros_like(student_logits)
                one_hot_x[
                    torch.arange(one_hot_x.shape[0], device=device)[:, None],
                    torch.arange(one_hot_x.shape[1], device=device)[None, :],
                    completion_ids
                ] = 1
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    student_logits.transpose(1, 2),
                    one_hot_x.transpose(1, 2)
                )
            else:
                raise NotImplementedError(self.loss_type)

            with torch.no_grad():
                student_predicted = student_logits.argmax(dim=2)
        else:
            raise NotImplementedError(self.loss_type)

        with torch.no_grad():
            acc_per_position = (completion_ids == student_predicted).float().mean(axis=0)
            accuracy = acc_per_position.mean()

        metrics = {
            'loss': loss, 
            'accuracy': accuracy,
            'acc_per_position': acc_per_position,
            'prompt_length': prompt_ids.shape[-1],
            'completion_length': completion_ids.shape[-1],
        }
        with torch.no_grad():
            selector = (
                torch.arange(student_logits.shape[0], device=device)[:, None],
                torch.arange(completion_ids.shape[-1], device=device)[None, :],
                completion_ids
            )

            # Perplexity computation
            student_nll = torch.nn.functional.log_softmax(student_logits, dim=-1)[selector]
            metrics['perp_per_position'] = torch.exp(student_nll.mean(dim=0))
            metrics['perp'] = torch.exp(student_nll.mean())
            if self.loss_type.startswith('kl'):
                tgt_logits = batch['logits']
                metrics['tgt_perp'] = torch.exp(torch.nn.functional.log_softmax(tgt_logits, dim=-1)[selector].mean())

            # Correct count before first error
            correct = (completion_ids == student_predicted).float().argmin(dim=1)
            # All correct => completion_length
            correct[(completion_ids == student_predicted).all(dim=1)] = completion_ids.shape[-1]
            metrics["correct"] = correct.float().mean()
        for positional_metric in ['acc_per_position', 'perp_per_position']:
            if positional_metric in metrics:
                short_positional_metric = positional_metric.replace('_per_position', '')
                for position in range(min(10, metrics[positional_metric].shape[0])):
                    metrics[f'{short_positional_metric}_pos_{position}'] = metrics[positional_metric][position]
        return metrics

    @torch.inference_mode()
    def generate_old(self, batch, max_new_tokens=None, max_length=None, attention_mask=None,
                 use_cache=True, return_metrics=False, do_sample=True, stopping_criteria=None,
                 pad_token_id=None, error_correction=False, eos=None, recompute=True, gated=False,
                 correcting_via = 'u', correct_via_hybrid=True, collect_stats=False):
        input_ids = batch['prompt_ids']
        assert input_ids.shape[0] == 1, "Batch size must be 1 for now"

        if max_length is not None:
            assert max_new_tokens is None, "Only one of max_length and max_new_tokens can be set"
            max_new_tokens = max_length - input_ids.shape[1]
        assert max_new_tokens is not None, "One of max_length and max_new_tokens must be set"
        assert max_new_tokens > 0, f"max_new_tokens must be positive, got {max_new_tokens}"
        if not do_sample:
            import warnings
            warnings.warn("do_sample=False is not supported, proceeding with sampling anyway.")

        # This would be faster using completion_ids and bin_edges but we want to measure the verification call.
        if error_correction:
            self.teacher.eval()
        prompt_ids = input_ids
        # prompt_ids = prompt_ids.repeat(20, 1)
        # if attention_mask is None:
        #     attention_mask = torch.ones_like(prompt_ids, dtype=torch.long, device=prompt_ids.device)
        # prompt_mask = attention_mask
        device = prompt_ids.device
        kv_student = None
        kv_teacher = None

        metrics = {
            'accuracy': [],
            'correct': [],
            'time_student': [],
            'time_teacher': [],
            'total_time': [],
            'token_per_time': []
        }
        # if all predicted tokens are correct, verification can add one extra token
        tokens_to_fill = max_new_tokens
        z_rnd_all = torch.rand([prompt_ids.shape[0], max_new_tokens + 1], device=device, dtype=torch.float32)
        if not recompute:
            left_bin_edges = batch['left_bin_edges']
            right_bin_edges = batch['right_bin_edges']
            z_rnd_all[:, :left_bin_edges.shape[1]] = left_bin_edges + (right_bin_edges - left_bin_edges) * z_rnd_all[:, :left_bin_edges.shape[1]]

        # Fill caches
        if use_cache and gated:
            self.student._gated_window = 0
            self.student._mode = 'teacher'
            outputs = self.student(
                input_ids=input_ids[:, :-1],
                use_cache=True,
                auxiliaries = z_rnd_all[:, 1000000:],
            )
            from copy import deepcopy
            kv_teacher = outputs.past_key_values
            kv_student = kv_teacher

        STP = {'x': [], 'y': [], 'n': [], 'nrel': []}
        reduced = self.tokens_per_student_call

        while tokens_to_fill > 0:
            # Student proposal
            n_prop = min(self.student_calls_per_step * self.tokens_per_student_call, tokens_to_fill)
            if collect_stats:
                n_prop = min(n_prop, reduced)
            z_idx = max_new_tokens - tokens_to_fill
            z_rnd = z_rnd_all[:, z_idx:z_idx + n_prop + 1]
            start_time = time.time()
            for k in range(min(self.student_calls_per_step, np.ceil(tokens_to_fill / self.tokens_per_student_call))):
                ths_n_prop = min((k + 1) * self.tokens_per_student_call,
                                 tokens_to_fill) - k * self.tokens_per_student_call
                if collect_stats:
                    ths_n_prop = min(ths_n_prop, reduced)
                # input_mask = torch.cat([
                #     prompt_mask,
                #     torch.ones([prompt_ids.shape[0], ths_n_prop], device=prompt_mask.device, dtype=prompt_mask.dtype)
                # ], dim=1)
                if gated:
                    self.student._gated_window = ths_n_prop
                    self.student._mode = 'student'
                if use_cache:
                    outputs = self.student(
                        input_ids=prompt_ids if kv_student is None else prompt_ids[:, kv_student.get_seq_length():],
                        # attention_mask=input_mask,
                        auxiliaries=z_rnd[
                            :, k * self.tokens_per_student_call:k * self.tokens_per_student_call + ths_n_prop],
                        past_key_values=kv_student,
                        use_cache=True
                    )
                    full_logits = outputs.logits
                    kv_student = outputs.past_key_values
                    if gated:
                        # pass
                        pos_pre = int(prompt_ids.shape[-1] - 1)
                        kv_student.crop(pos_pre)
                        kv_teacher = kv_student
                else:
                    full_logits = self.student(
                        input_ids=prompt_ids,
                        # attention_mask=input_mask,
                        auxiliaries=z_rnd[
                            :, k * self.tokens_per_student_call:k * self.tokens_per_student_call + ths_n_prop],
                    ).logits
                if self.loss_type.startswith('kl'):
                    raise NotImplementedError
                elif 'bc' in self.loss_type:
                    # ------
                    # O-PTP
                    # ------
                    student_logits = full_logits[:, -ths_n_prop:]
                    student_predicted = student_logits.argmax(dim=2)
                    # gated guarantees one token
                    if gated:
                        tgt_logits = self.adapt_logits(full_logits[:, -ths_n_prop - 1])
                        right_bin_edges = torch.softmax(tgt_logits, dim=-1).cumsum(dim=-1)
                        right_bin_edges[..., -1] = 1
                        right_bin_edges[..., 0] = 0
                        correct_token = (right_bin_edges > z_rnd[:, :1][..., None]).max(dim=-1).indices
                        student_predicted[:, 0] = correct_token[0, 0]
                    student_completion = torch.cat([
                        prompt_ids,
                        student_predicted,
                    ], dim=1)
                else:
                    raise NotImplementedError(self.loss_type)
                prompt_ids = student_completion

                # prompt_mask = input_mask
            # torch.cuda.synchronize()
            metrics['time_student'] += [time.time() - start_time]

            # Skip teacher
            if correcting_via == 'true':
                num_correct = ths_n_prop
                accuracy = 1.0
                metrics['accuracy'] += [accuracy]
                metrics['correct'] += [num_correct]
                pos_pre = int(prompt_ids.shape[-1] - n_prop)
                # prune kv-cache
                if kv_student is not None and not gated:
                    kv_student.crop(pos_pre)
                tokens_to_fill -= num_correct
                if eos in prompt_ids[0, pos_pre:]:
                    break
                continue

            # Teacher verification
            start_time = time.time()
            if gated:
                self.student._gated_window = 0
                self.student._mode = 'teacher'
            if use_cache:
                # if kv_teacher is None and kv_student is not None:
                #     from copy import deepcopy
                #     kv_teacher = deepcopy(kv_student)
                #     kv_teacher.crop(input_ids.shape[1])
                if gated:
                    outputs = self.student(
                        auxiliaries = z_rnd[:, 1000000:],
                        input_ids=prompt_ids if kv_teacher is None else prompt_ids[:, kv_teacher.get_seq_length():],
                        # attention_mask=prompt_mask,
                        past_key_values=kv_teacher,
                        use_cache=True
                    )
                else:
                    outputs = self.teacher(
                        input_ids=prompt_ids if kv_teacher is None else prompt_ids[:, kv_teacher.get_seq_length():],
                        # attention_mask=prompt_mask,
                        past_key_values=kv_teacher,
                        use_cache=True
                    )
                kv_teacher = outputs.past_key_values
            else:
                # outputs = self.student(
                outputs = self.teacher(
                    input_ids=prompt_ids,
                    # attention_mask=prompt_mask,
                    return_scores=True
                )

            tgt_logits = outputs.logits[..., - n_prop - 1:, :]
            tgt_logits = self.adapt_logits(tgt_logits)
            # ? assert logits.dtype == torch.float32, logits.dtype

            # Our style
            right_bin_edges = torch.softmax(tgt_logits, dim=-1).cumsum(dim=-1)
            right_bin_edges[..., -1] = 1
            right_bin_edges[..., 0] = 0
            correct_tokens = (right_bin_edges > z_rnd[..., None]).max(dim=-1).indices
            if correcting_via in ['threshold', 'threshold-q']:
                assert correct_via_hybrid
                pred_tokens = prompt_ids[..., - n_prop:][..., :correct_tokens.shape[1]]
                if correcting_via == 'threshold':
                    tgt_probs = torch.softmax(tgt_logits, dim=-1)[:, :-1].gather(dim=2, index=pred_tokens.unsqueeze(-1)).squeeze(-1)
                else:
                    tgt_probs = torch.softmax(student_logits, dim=-1).gather(dim=2, index=pred_tokens.unsqueeze(-1)).squeeze(-1)
                correct_mask = (0.8 <= tgt_probs)
                correct_tokens[..., :-1][correct_mask] = pred_tokens[correct_mask].clone()
            elif correcting_via in ['dist', 'dist-q']:
                # Spec-style MH step
                pred_tokens = prompt_ids[..., - n_prop:][..., :correct_tokens.shape[1]]
                if correct_via_hybrid:
                    correct_mask = correct_tokens[..., :-1] == pred_tokens
                else:
                    correct_mask = torch.ones_like(pred_tokens) == 0
                tgt_probs = torch.softmax(tgt_logits, dim=-1)[:, :-1].gather(dim=2, index=pred_tokens.unsqueeze(-1)).squeeze(-1)
                if correcting_via == 'dist':
                    r = torch.rand_like(tgt_probs)
                    correct_mask = (r <= tgt_probs) | correct_mask
                else:
                    std_probs = torch.softmax(student_logits, dim=-1).gather(dim=2, index=pred_tokens.unsqueeze(-1)).squeeze(-1)
                    r = torch.rand_like(tgt_probs)
                    correct_mask = (std_probs <= tgt_probs) | (r <= tgt_probs/std_probs) | correct_mask
                correct_tokens[..., :-1] = pred_tokens.clone()
                if not correct_mask.all():
                    if correcting_via == 'dist':
                        new_logits = tgt_logits.clone()
                        new_logits.scatter_(dim=2, index=pred_tokens.unsqueeze(-1), value=float('-inf'))
                    else:
                        new_probs = torch.softmax(tgt_logits, dim=-1).clone()
                        reduced = (tgt_probs - std_probs).clip(min=0)
                        new_probs.scatter_(dim=2, index=pred_tokens.unsqueeze(-1), src=reduced.unsqueeze(-1))
                        new_logits = torch.log(new_probs / new_probs.sum(dim=-1, keepdim=True) + 1e-8)
                    correct_tokens[..., :-1][~correct_mask] = torch.distributions.Categorical(logits=new_logits[:, :-1][~correct_mask]).sample()
            # elif correcting_via == 'true':
            #     correct_tokens[..., :3] = pred_tokens[:3].clone()
            elif correcting_via != 'u':
                raise NotImplementedError

            # print(correct_tokens.min(), correct_tokens.max())
            # print('a')
            # correct_right = torch.gather(right_bin_edges, dim=2, index=correct_tokens[..., None])[..., 0]
            # correct_left = torch.gather(right_bin_edges, dim=2, index=correct_tokens[..., None]-1)[..., 0]
            # print('b')
            # pos_in_interval = (z_rnd - correct_left) / (correct_right - correct_left)
            if not recompute:
                correct_tokens = batch['completion_ids'][:, z_idx:z_idx + n_prop]
            if self.student_calls_per_step >= 1:
                if not recompute:
                    check_tokens = correct_tokens
                else:
                    check_tokens = correct_tokens[..., :-1]
                # predict_tokens = prompt_ids[..., - n_prop:]
                predict_tokens = prompt_ids[..., - n_prop:][..., :correct_tokens.shape[1]]
                num_correct = (check_tokens == predict_tokens).float().argmin(dim=1)
                num_correct[(check_tokens == predict_tokens).all(dim=1)] = n_prop
                if collect_stats:
                    student_p = torch.softmax(student_logits, dim=-1)[0, range(ths_n_prop), student_predicted]
                    STP['y'] += [(check_tokens == predict_tokens).detach().cpu()]
                    STP['x'] += [(student_p).detach().cpu()]
                    # Estimated correctness likelihood
                    # A = np.insert(np.cumprod(student_p), 0, 1) * (1 - np.append(student_p, 0))
                    A = torch.cumprod(student_p[0], dim=-1)
                    A[:-1] *= 1 - student_p[0, 1:]
                    # Optimize as Knapsack
                    if False:
                        W = A * torch.arange(1, ths_n_prop + 1, device=A.device)
                        idxs = W.argsort(descending=True)
                        slowed = ths_n_prop / 50 * torch.arange(1, ths_n_prop + 1, device=W.device)
                        K = torch.argmax(torch.cumsum(W[idxs], dim=-1) / (1 + slowed)) + 1
                        B = torch.tensor([ths_n_prop if i in idxs[:K] else 0 for i in range(ths_n_prop + 1)], device=A.device)
                    # Optimize with Dinkelbach
                    else:
                        Y = np.load('data/Y.npy')
                        YY = Y.argmin(axis=1)
                        YY[Y.all(axis=1)] = 20
                        # Dirichilet prior with effective sample size = 50
                        base = np.unique(YY, return_counts=True)[1] / YY.shape[0] * 50
                        diri = np.zeros(21)
                        # if len(np.array(metrics['correct'])[np.array(metrics['correct']) > 0][np.array(STP['nrel']) < 1]) > 0:
                        #     diri += np.bincount(np.array(metrics['correct'])[np.array(metrics['correct']) > 0][np.array(STP['nrel']) < 1], minlength=21)
                        # for th in np.array(metrics['correct'])[np.array(metrics['correct']) > 0][np.array(STP['nrel']) == 1]:
                        #     diri[th+1:] += base[th+1:] / base[th+1:].sum()
                        #     # diri[th + 1:] += 1 # / base[th+1:].shape[0]
                        H = torch.tensor(np.cumsum((base + diri) / (base + diri).sum() * np.arange(21)), device=A.device)
                        if len(STP['nrel']) > 0 and np.array(STP['nrel']).mean() > 0.2:
                            new_lam = np.clip((np.array(STP['nrel']).mean() - 0.2) / 0.2, 0, 1)
                            H = (1 - new_lam) * H + new_lam * torch.arange(self.tokens_per_student_call + 1, device=A.device)
                        # H = torch.arange(self.tokens_per_student_call + 1, device=A.device)
                        lam_pre, lam = 0, 0
                        for _ in range(20):
                            B = torch.argmax(A[:, None] * H[None, :self.tokens_per_student_call + 1] - lam * torch.arange(self.tokens_per_student_call + 1, device=A.device)[None, :] / 50, dim=1)
                            lam_pre = lam
                            lam = torch.sum(A * H[B]) / (1 + B.sum() / 50)
                            if (lam_pre == lam).all():
                                break
                    # todo, grab lower in case of miss
                    reduced = int(B[num_correct - 1])
                    STP['n'] += [sum(B).item()]
                    STP['nrel'] += [(num_correct / ths_n_prop).item()]
                    if reduced == 0:
                        metrics['accuracy'] += [0]
                        metrics['correct'] += [0]
                        reduced = self.tokens_per_student_call

                idx = 0 # int(torch.abs(pos_in_interval - 0.5)[:, 0].argmin())
                num_correct = min(int(num_correct[idx].float().mean()), 15)
                accuracy = (check_tokens == predict_tokens)[idx].float().mean().item()
            else:
                num_correct = 0
                accuracy = 1.0
            metrics['accuracy'] += [accuracy]
            metrics['correct'] += [num_correct + 1]

            # Fix first mistake, only works for batch_size=1 for now
            pos = int(prompt_ids.shape[-1] - (n_prop - num_correct))
            pos_pre = int(prompt_ids.shape[-1] - n_prop)
            if n_prop > num_correct:
                prompt_ids = prompt_ids[idx, :pos].expand(prompt_ids.shape[0], -1)
                # prompt_mask = prompt_mask[..., :pos]
            # prune kv-cache
            if kv_teacher is not None:
                kv_teacher.crop(pos)
                # for l in kv_teacher.layers:
                #     l.keys = l.keys[idx, ...].expand(prompt_ids.shape[0], *[-1] * 3)
                #     l.values = l.values[idx, ...].expand(prompt_ids.shape[0], *[-1] * 3)
            if kv_student is not None and not gated:
                kv_student.crop(pos_pre)
                # for l in kv_student.layers:
                #     l.keys = l.keys[idx, ...].expand(prompt_ids.shape[0], *[-1] * 3)
                #     l.values = l.values[idx, ...].expand(prompt_ids.shape[0], *[-1] * 3)
            prompt_ids = torch.cat([
                prompt_ids,
                correct_tokens[idx:idx+1, num_correct][:, None].repeat(prompt_ids.shape[0], 1),
            ], dim=1)
            # prompt_mask = torch.cat([
            #     prompt_mask,
            #     torch.ones([1, 1], device=prompt_mask.device, dtype=prompt_mask.dtype),
            # ], dim=1)
            tokens_to_fill -= (num_correct + 1)

            # torch.cuda.synchronize()
            metrics['time_teacher'] += [time.time() - start_time]

            if eos in prompt_ids[idx, pos_pre:]:
                break
            # if stopping_criteria is not None and stopping_criteria(prompt_ids):
            #     break

        # import matplotlib.pyplot as plt; from sklearn.linear_model import LogisticRegression; X = torch.concat(STP['x'], dim=1).cpu()[0]; Y = torch.concat(STP['y'], dim=1).cpu()[0]; clf = LogisticRegression(); clf.fit(X[:, None], Y)
        # plt.scatter(X, Y, alpha=0.02); plt.plot(np.linspace(0, 1, 100), clf.predict_proba(np.linspace(0, 1, 100)[:, None])[:, 1], label='LogReg'); plt.xlabel('Student confidence'); plt.ylabel('Teacher Verification')
        # plt.plot([i / 10 for i in range(11) for _ in range(2)][1:-1], [Y[torch.floor(X * 10) == i].float().mean() for i in range(10) for _ in range(2)], label='binned'); plt.legend(); plt.tight_layout(); plt.show()

        total_time = sum(metrics['time_student'] + metrics['time_teacher'])
        metrics = {
            'completion': prompt_ids,
            'accuracy': np.mean(metrics['accuracy']),
            'correct_per_call': np.mean(metrics['correct']) - 1,
            'correct_first': metrics['correct'][0] - 1,
            # 'correct_second': metrics['correct'][1] - 1,
            'correct_all': metrics['correct'],
            'time_student': np.mean(metrics['time_student']),
            'time_teacher': np.mean(metrics['time_teacher']),
            'time': total_time,
            'num_calls': len(metrics['correct']),
            'token_per_time': (sum(metrics['correct'])) / total_time,
            'STP': STP
        }

        if return_metrics:
            return input_ids, metrics
        return input_ids

    # @torch.inference_mode()
    # def generate(self, batch, max_new_tokens, use_cache=True, return_metrics=False, eos=None, recompute=True,
    #              gated=True, correcting_via=None, collect_stats=None):
    #     prompt_ids = batch['prompt_ids']
    #     assert prompt_ids.shape[0] == 1, "Batch size must be 1 for now"
    #
    #     # if attention_mask is None:
    #     #     attention_mask = torch.ones_like(prompt_ids, dtype=torch.long, device=prompt_ids.device)
    #     # prompt_mask = attention_mask
    #     device = prompt_ids.device
    #     kv_cache = DynamicCache()
    #
    #     metrics = {
    #         'accuracy': [],
    #         'correct': [],
    #         'time': [],
    #         'time-1': [],
    #         'time-2': [],
    #         'time-3': [],
    #         'time-4': [],
    #         'time_model': [],
    #     }
    #     # if all predicted tokens are correct, verification can add one extra token
    #     tokens_to_fill = max_new_tokens
    #     tokens_to_verify = max_new_tokens
    #
    #     z_rnd_all = torch.rand([prompt_ids.shape[0], max_new_tokens + self.tokens_per_student_call], device=device, dtype=torch.float32)
    #
    #     while tokens_to_verify > 0:
    #         # Student proposal
    #         n_prop = max(min(self.tokens_per_student_call, tokens_to_fill), 0)
    #         n_verify = tokens_to_verify - tokens_to_fill
    #         z_idx = max_new_tokens - tokens_to_fill
    #         # z_rnd = z_rnd_all[:, z_idx:z_idx + n_prop]
    #         z_rnd = torch.cat([z_rnd_all[:, z_idx - d:z_idx - d + n_prop] for d in range(n_verify + 1)], dim=1)
    #
    #         input_ids = prompt_ids
    #         if not gated:
    #             z_rnd = z_rnd.repeat(2, 1)
    #             input_ids = input_ids.repeat(2, 1)
    #         else:
    #             self.student._gated_window = (n_verify + 1) * n_prop
    #         # seq_len = input_ids.shape[1] + n_prop
    #         # todo: could be more if n_prop < tokens_per_student_call
    #         seq_len = prompt_ids.shape[1] + (n_verify + 1) * n_prop
    #         input_mask = torch.tril(torch.ones(seq_len, seq_len))
    #         for d in range(1, n_verify+1):
    #             input_mask[-d * n_prop: seq_len - (d-1) * n_prop, input_ids.shape[1] - n_verify - 1 + d:-d * n_prop] = 0
    #         input_mask = input_mask[kv_cache.get_seq_length():]
    #         input_mask = (1 - input_mask[None, None, :, :].to('cuda').to(torch.float16)) * -1e15
    #         input_position_ids = torch.arange(kv_cache.get_seq_length(), seq_len, device=input_ids.device)[None, :]
    #         pos = input_ids.shape[1] - kv_cache.get_seq_length()
    #         for d in range(0, n_verify + 1):
    #             input_position_ids[:, pos + d * n_prop: pos + (d+1) * n_prop] -= d * n_prop - d + 1
    #         outputs = self.student(
    #             input_ids=input_ids[:, kv_cache.get_seq_length():],
    #             attention_mask=input_mask,
    #             position_ids=input_position_ids,
    #             auxiliaries=z_rnd,
    #             past_key_values=kv_cache,
    #             use_cache=True
    #         )
    #         kv_cache = outputs.past_key_values
    #         full_logits = outputs.logits
    #         # tgt_logits = full_logits[-1:, :-n_prop]
    #         tgt_logits = full_logits[-1:, :pos]
    #         # O-PTP
    #         assert 'bc' in self.loss_type
    #         # student_logits = full_logits[:1, -n_prop:]
    #         student_logits = full_logits[:1, pos:]
    #         student_predicted = student_logits.argmax(dim=-1)
    #
    #         # Verify last speculated tokens
    #         if n_verify > 0:
    #             assert n_verify == tgt_logits.shape[1] - 1
    #             tgt_logits = self.adapt_logits(tgt_logits)
    #             right_bin_edges = torch.softmax(tgt_logits, dim=-1).cumsum(dim=-1)
    #             right_bin_edges[..., -1] = 1
    #             right_bin_edges[..., 0] = 0
    #             z_idx = max_new_tokens - tokens_to_verify
    #             z_rnd = z_rnd_all[:, z_idx:z_idx + n_verify + 1]
    #             correct_tokens = (right_bin_edges > z_rnd[..., None]).max(dim=-1).indices
    #             check_tokens = correct_tokens[:, :-1]
    #             predict_tokens = prompt_ids[..., - n_verify:]
    #             matches = (predict_tokens == check_tokens)
    #             num_correct = matches.float().argmin(dim=1)
    #             num_correct[matches.all(dim=1)] = n_verify
    #             num_correct = int(num_correct[0])
    #             # num_new = max(num_correct, 1)
    #             num_new = num_correct + 1
    #             # accuracy = matches[0].float().mean().item()
    #             # tokens_to_verify -= num_new
    #             tokens_to_verify -= num_correct
    #             kv_cache.crop(int(prompt_ids.shape[-1] - (n_verify - num_correct)))
    #             # if num_correct < n_verify:
    #             #     prompt_ids = torch.cat([
    #             #         prompt_ids[:, :-(n_verify - num_correct)],
    #             #         correct_tokens[:, :num_new],
    #             #     ], dim=1)
    #             #     tokens_to_fill = tokens_to_verify
    #             #     metrics['correct'] += [num_new]
    #             # else:
    #             if True:
    #                 #
    #                 student_predicted = student_predicted[:, -(num_correct + 1) * n_prop: student_predicted.shape[-1] - num_correct * n_prop]
    #                 # Verify first speculated and add other speculated tokens
    #                 match = student_predicted[0, 0] == correct_tokens[0, num_correct] if n_prop > 0 else False
    #                 if not match:
    #                     # assert not gated
    #                     # Discard speculated tokens
    #                     prompt_ids = torch.cat([
    #                         prompt_ids[:, :prompt_ids.shape[1] - n_verify],
    #                         correct_tokens[:, :num_correct+1],
    #                     ], dim=1)
    #                     tokens_to_verify -= 1
    #                     tokens_to_fill = tokens_to_verify
    #                 else:
    #                     # Add new speculated tokens
    #                     prompt_ids = torch.cat([
    #                         prompt_ids[:, :prompt_ids.shape[1] - (n_verify - num_correct)],
    #                         student_predicted,
    #                     ], dim=1)
    #                     tokens_to_fill = tokens_to_verify - n_prop
    #                     tokens_to_verify -= 1
    #                 metrics['correct'] += [num_new]
    #             # metrics['accuracy'] += [accuracy]
    #             if eos in prompt_ids[0, -num_new:]:
    #                 break
    #         else:
    #             # Accept one token
    #             tgt_logits = self.adapt_logits(tgt_logits[:, -1:])
    #             right_bin_edges = torch.softmax(tgt_logits, dim=-1).cumsum(dim=-1)
    #             right_bin_edges[..., -1] = 1
    #             right_bin_edges[..., 0] = 0
    #             z_idx = max_new_tokens - tokens_to_verify
    #             z_rnd = z_rnd_all[:, z_idx:z_idx + 1]
    #             correct_token = (right_bin_edges > z_rnd[..., None]).max(dim=-1).indices
    #             match = student_predicted[0, 0] == correct_token[0, 0]
    #             kv_cache.crop(int(prompt_ids.shape[-1]))
    #             if not match:
    #                 # assert not gated
    #                 # Discard speculated tokens
    #                 prompt_ids = torch.cat([
    #                     prompt_ids,
    #                     correct_token,
    #                 ], dim=1)
    #                 tokens_to_verify -= 1
    #                 tokens_to_fill = tokens_to_verify
    #                 # metrics['accuracy'] += [0]
    #             else:
    #                 # Add new speculated tokens
    #                 prompt_ids = torch.cat([
    #                     prompt_ids,
    #                     student_predicted,
    #                 ], dim=1)
    #                 tokens_to_fill = tokens_to_verify - n_prop
    #                 tokens_to_verify -= 1
    #                 # metrics['accuracy'] += [1]
    #             metrics['correct'] += [1]
    #
    #     metrics = {
    #         'completion': prompt_ids,
    #         # 'accuracy': np.mean(metrics['accuracy']),
    #         'correct_per_call': np.mean(metrics['correct']),
    #         'correct_all': metrics['correct'],
    #         'num_calls': len(metrics['correct']),
    #     }
    #
    #     if return_metrics:
    #         return input_ids, metrics
    #     return input_ids

    def proposals(self, num_tokens=None, student_p=None, n_verify=None, double_at=100, metrics=None):
        """
        Optimize proposals B wrt overhead adjusted expected # correct tokens
        max_B [sum_i A_i * H(B_i)] / [1 + sum_i B_i / 50)]

        If A_0 = 1 this becomes max_k H(k) / [1 + k / 50] = 14
        """
        # Estimated probability of # correct tokens
        if student_p is None:
            A = self.hist_base
            if n_verify is not None:
                A = torch.cat([A[:n_verify], torch.tensor([A[n_verify:].sum()])])
        else:
            # assert student_p.shape[1] == n_verify
            A = torch.ones(student_p.shape[1] + 1)
            A[1:] = torch.cumprod(student_p[0].cpu(), dim=-1)
            A[:-1] *= 1 - student_p[0].cpu()
        # Reward; Estimated # correct tokens in the next step given k proposed tokens
        H_hist = torch.cumsum(self.hist_base * self.arange_21, dim=-1)
        # or: # proposed tokens
        H_count = self.arange_21
        # switch dynamically
        # if metrics is not None and len(metrics['Nrel']) > 0:
        #     rho = np.clip((np.array(metrics['Nrel']).mean() - 0.2) / 0.2, 0, 1)
        # else:
        #     rho = 0
        rho = 0
        H = (1 - rho) * H_hist + rho * H_count
        # A = A.clip(min=0.05)

        M = self.tokens_per_student_call
        AH = A[:, None] * H[None, :M + 1]  # [n_pos, M+1]

        if num_tokens is None:
            # Optimize based on per-token cost of 1/<double_at>
            lam = 0
            for _ in range(3):
                B = torch.argmax(AH - lam * self.arange_tpscp1[None, :] / double_at, dim=1)
                lam_pre = lam
                lam = torch.sum(A * H[B]) / (1 + B.sum() / double_at)
                if (lam_pre == lam).all():
                    break
            return B.tolist()
        else:
            num_tokens = min(num_tokens, M * A.shape[0])

            # --- Binary search on lambda --- faster at first
            # lam_lo = 0.0
            # lam_hi = float(A.max())
            # for _ in range(50):
            #     lam = (lam_lo + lam_hi) / 2
            #     B = torch.argmax(AH - lam * self.arange_tpscp1[None, :], dim=1)
            #     total = B.sum().item()
            #     if total > num_tokens:
            #         lam_lo = lam
            #     elif total < num_tokens:
            #         lam_hi = lam
            #     else:
            #         break
            #     if abs(total - num_tokens) <= 5:
            #         break

            # Greedy estimate of lam
            A_idx = torch.argsort(A, descending=True)
            R = num_tokens // M
            r = num_tokens % M
            B = torch.zeros([A.shape[0]], dtype=int)
            B[A_idx[:R]] = M
            B[A_idx[R]] = r
            if r > 0:
                lam = float(A[A_idx[R]] * (H[r] - H[r - 1]))
            else:
                lam = float(A[A_idx[R - 1]] * (H[M] - H[M - 1]))
            B = torch.argmax(AH - lam * self.arange_tpscp1[None, :], dim=1)
            print(B.sum())

            # --- Greedy correction ---
            # B = B.clone()
            total = B.sum().item()

            while total > num_tokens:
                # Decrement the position with the smallest marginal gain of its last token
                # marginal gain of token b_i: A_i * (H[b_i] - H[b_i-1])
                can_dec = B > 0
                gains = torch.where(can_dec, A * (H[B] - H[(B - 1).clamp(min=1)]), self.inf)
                idx = torch.argmin(gains)
                B[idx] -= 1
                total -= 1

            while total < num_tokens:
                # Increment the position with the highest marginal gain of the next token
                # marginal gain of token b_i+1: A_i * (H[b_i+1] - H[b_i])
                can_inc = B < M
                gains = torch.where(can_inc, (A * (H[(B + 1).clamp(max=M)] - H[B])).clamp(min=1e-10), -self.inf)
                idx = torch.argmax(gains)
                B[idx] += 1
                total += 1

            return B.tolist()

    @torch.inference_mode()
    def generate(self, batch, max_new_tokens, return_metrics=False, eos=None,
                 fixed_tokens=True, pad_token=13, **kwargs):
        """
        Efficient Quadratic Coding using kv-cached Gated LoRA

        Input:
        ------
        fixed_tokens - if true force each transformer call to have the same number of verifying tokens and proposed
                       tokens, padding if necessary.
        """
        # todo remove syncs
        prompt_ids = batch['prompt_ids']
        assert prompt_ids.shape[0] == 1, "Batch size must be 1"
        assert self.student.shift_positions
        assert 'bc' in self.loss_type
        assert self.top_k is not None and self.top_k > 0
        assert self.top_p is not None and self.top_p < 1.0
        metrics = {'correct': [], 'N': [], 'Nrel': []}
        # timing = {'step': [], 'call': []}
        # import time
        # from torch.profiler import record_function

        # If we use a fixed number of proposed tokens, the Gated LoRA layers are already set
        num_proposed_tokens = GatedLinearLora.GATE_WINDOW if fixed_tokens else None

        # Verify in parallel
        tokens_to_fill = max_new_tokens
        tokens_to_verify = max_new_tokens
        z_rnd_all = torch.rand([prompt_ids.shape[0], max_new_tokens + self.tokens_per_student_call + (num_proposed_tokens if fixed_tokens else 0)], device=device, dtype=torch.float32)
        if fixed_tokens:
            n_props = ((num_proposed_tokens // self.tokens_per_student_call) * [self.tokens_per_student_call] + [(num_proposed_tokens % self.tokens_per_student_call)] + self.tokens_per_student_call * [0])[:self.tokens_per_student_call]
        else:
            n_props = self.proposals(n_verify=0, num_tokens=num_proposed_tokens)

        # Fill caches
        if self.past_kv_cache is not None:
            past_prompt_ids, past_kv_cache = self.past_kv_cache
            end = min(prompt_ids.shape[1], past_prompt_ids.shape[1])
            match = prompt_ids[:, :end] == past_prompt_ids[:, :end]
            keep = end if match.all() else match.float().argmin().item()
            # prompt_ids[:, :keep] = past_prompt_ids[:, :keep]
            past_kv_cache.crop(keep)
            kv_cache = past_kv_cache
        else:
            kv_cache = DynamicCache()
            # kv_cache = StaticCache()
        # scall = time.time()
        if not fixed_tokens:
            self.student.set_gate_window(0)
        # todo optimize teacher time
        outputs = self.student(
            input_ids=prompt_ids[:, kv_cache.get_seq_length():-1],
            auxiliaries=z_rnd_all[:, :num_proposed_tokens if fixed_tokens else 0],
            past_key_values=kv_cache,
            use_cache=True,
        )
        kv_cache = outputs.past_key_values
        if fixed_tokens:
            # Pad to length
            kv_cache.crop(prompt_ids.shape[1] - 1)
            prompt_ids = torch.cat([
                prompt_ids,
                pad_token * self.ones_tpsc[:, :self.tokens_per_student_call - 1]
            ], dim=1)
            tokens_to_fill -= self.tokens_per_student_call - 1
        # torch.cuda.synchronize()
        # timing['call'] += [time.time() - scall]

        while tokens_to_verify > 0:
            if not fixed_tokens:
                n_props = [min(n, max(0, tokens_to_verify - d)) for d, n in enumerate(n_props)]
            n_verify = tokens_to_verify - tokens_to_fill
            # assert n_verify == len(n_props) - 1
            seq_len = prompt_ids.shape[1] + sum(n_props)
            pos = prompt_ids.shape[1] - kv_cache.get_seq_length()
            metrics['N'] += [pos + sum(n_props)]

            # Prepare inputs
            z_idx = max_new_tokens - tokens_to_verify
            z_rnd = torch.cat([z_rnd_all[:, z_idx + d:z_idx + d + n_prop] for d, n_prop in enumerate(n_props)], dim=1)
            input_ids = prompt_ids[:, kv_cache.get_seq_length():]
            midx = pos
            # todo pre-allocate buffer space for inputs
            input_mask = torch.tril(torch.ones(seq_len - kv_cache.get_seq_length(), seq_len, device=device), diagonal=kv_cache.get_seq_length())
            input_position_ids = torch.arange(kv_cache.get_seq_length(), seq_len, device=device)[None, :]
            for d, n_prop in enumerate(n_props):
                input_mask[midx: midx + n_prop, prompt_ids.shape[1] - n_verify + d:kv_cache.get_seq_length() + midx] = 0
                input_position_ids[:, midx: midx + n_prop] -= midx - pos + n_verify - d + 1
                midx += n_prop
            input_mask = (1 - input_mask[None, None, :, :].to(torch.float16)) * -1e15

            # Student proposals
            if not fixed_tokens:
                self.student.set_gate_window(sum(n_props))
            outputs = self.student(
                input_ids=input_ids,
                attention_mask=input_mask,
                position_ids=input_position_ids,
                auxiliaries=z_rnd,
                past_key_values=kv_cache,
                use_cache=True
            )

            kv_cache = outputs.past_key_values
            full_logits = outputs.logits
            student_logits = full_logits[:, pos:]
            if self.temperature is not None and self.temperature != 1.0:
                full_logits[:, :pos] = full_logits[:, :pos] / self.temperature
            full_p = torch.softmax(full_logits, dim=-1)
            student_p = full_p[:, pos:]
            tgt_p = self.adapt_p(full_p[:, :pos])
            student_predicted = student_logits.argmax(dim=-1)
            student_p_max = student_p.gather(-1, student_predicted[..., None])[..., 0]

            # Verify last speculated tokens
            if n_verify > 0:
                # assert n_verify == tgt_logits.shape[1] - 1
                # with record_function("loop: verify + accept"):
                right_bin_edges = tgt_p.cumsum(dim=-1)
                right_bin_edges[..., -1] = 1
                z_idx = max_new_tokens - tokens_to_verify
                z_rnd = z_rnd_all[:, z_idx:z_idx + n_verify + 1]
                correct_tokens = (right_bin_edges > z_rnd[..., None]).max(dim=-1).indices
                check_tokens = correct_tokens[:, :-1]
                predict_tokens = prompt_ids[..., - n_verify:]
                matches = (predict_tokens == check_tokens)
                num_correct = matches.float().argmin(dim=1)
                num_correct[matches.all(dim=1)] = n_verify
                num_correct = int(num_correct[0])
                num_new = num_correct + 1
                tokens_to_verify -= num_correct
                kv_cache.crop(prompt_ids.shape[-1] - (n_verify - num_correct))
                prev_prop = sum(n_props[:num_correct])
                ths_student_predicted = student_predicted[:, prev_prop: prev_prop + n_props[num_correct]]
                ths_student_p = student_p_max[:, prev_prop: prev_prop + n_props[num_correct]]
                metrics['Nrel'] += [num_correct / n_verify]
                # Verify first speculated and add other speculated tokens
                if fixed_tokens and ths_student_predicted.shape[1] < self.tokens_per_student_call:
                    # Pad with linbebreaks
                    ths_student_predicted = torch.cat([ths_student_predicted, pad_token * self.ones_tpsc[:, :self.tokens_per_student_call - ths_student_predicted.shape[1]]], dim=1)
                    ths_student_p = torch.cat([ths_student_p, 0 * self.ones_tpsc[:, :self.tokens_per_student_call - ths_student_p.shape[1]]], dim=1)
                if ths_student_predicted.shape[1] == 0:
                    match = False
                else:
                    ths_student_predicted[0, 0] = correct_tokens[0, num_correct]
                    match = ths_student_predicted[0, 0] == correct_tokens[0, num_correct]
                if not match:
                    # Discard speculated tokens
                    prompt_ids = torch.cat([
                        prompt_ids[:, :prompt_ids.shape[1] - n_verify],
                        correct_tokens[:, :num_correct + 1],
                    ], dim=1)
                    tokens_to_verify -= 1
                    tokens_to_fill = tokens_to_verify
                    n_props = self.proposals(n_verify=0, num_tokens=num_proposed_tokens, metrics=metrics)
                else:
                    # Add new speculated tokens
                    prompt_ids = torch.cat([
                        prompt_ids[:, :prompt_ids.shape[1] - (n_verify - num_correct)],
                        ths_student_predicted,
                    ], dim=1)
                    tokens_to_fill = tokens_to_verify - ths_student_predicted.shape[1]
                    tokens_to_verify -= 1
                    n_props = self.proposals(student_p=ths_student_p[:, 1:], num_tokens=num_proposed_tokens, metrics=metrics)
                metrics['correct'] += [num_new]
                if eos in correct_tokens[:, :num_correct + 1]:
                    break
            else:
                # Only relevant if fixed_tokens=False and we don't force correct tokens
                raise NotImplementedError
            # else:
            #     # Accept one token
            #     tgt_logits = self.adapt_logits(tgt_logits[:, -1:])
            #     right_bin_edges = torch.softmax(tgt_logits, dim=-1).cumsum(dim=-1)
            #     right_bin_edges[..., -1] = 1
            #     right_bin_edges[..., 0] = 0
            #     z_idx = max_new_tokens - tokens_to_verify
            #     z_rnd = z_rnd_all[:, z_idx:z_idx + 1]
            #     correct_token = (right_bin_edges > z_rnd[..., None]).max(dim=-1).indices
            #     kv_cache.crop(int(prompt_ids.shape[-1]))
            #
            #     if fixed_tokens and student_predicted.shape[1] < self.tokens_per_student_call:
            #         # Pad with linbebreaks
            #         student_predicted = torch.cat([student_predicted, 13 * torch.ones([1, self.tokens_per_student_call - ths_student_predicted.shape[1]], dtype=int, device=device)], dim=1)
            #         student_p_max = torch.cat([student_p_max, 13 * torch.ones([1, self.tokens_per_student_call - ths_student_p.shape[1]], device=device)], dim=1)
            #     if student_predicted.shape[1] == 0:
            #         match = False
            #     else:
            #         student_predicted[0, 0] = correct_token[0, 0]
            #         match = student_predicted[0, 0] == correct_token[0, 0]
            #     if not match:
            #         # Discard speculated tokens
            #         prompt_ids = torch.cat([
            #             prompt_ids,
            #             correct_token,
            #         ], dim=1)
            #         tokens_to_verify -= 1
            #         tokens_to_fill = tokens_to_verify
            #         n_props = self.proposals(n_verify=0, num_tokens=num_proposed_tokens, metrics=metrics)
            #     else:
            #         # Add new speculated tokens
            #         prompt_ids = torch.cat([
            #             prompt_ids,
            #             student_predicted[:, :n_props[0]],
            #         ], dim=1)
            #         tokens_to_fill = tokens_to_verify - n_props[0]
            #         tokens_to_verify -= 1
            #         n_props = self.proposals(student_p=student_p_max[:, 1:], num_tokens=num_proposed_tokens, metrics=metrics)
            #     metrics['correct'] += [1]
            #     if eos == correct_token[0, 0]:
            #         break

            # torch.cuda.synchronize()
            # timing['step'] += [time.time() - s]

        # print(1000 * np.mean(timing['call'][1:]), 1000 * timing['call'][0], 1000 * np.mean(timing['step']))
        # plt.scatter(metrics['off'], metrics['offp'], s=2, alpha=0.5); plt.gca().set(xlabel='If the predicted token is wrong, the k-th one after is', ylabel='student confidence for actual correct token'); plt.show()
        metrics = {
            'completion': prompt_ids,
            'correct_per_call': np.mean(metrics['correct']),
            'correct_all': metrics['correct'],
            'num_calls': len(metrics['correct']),
        }

        self.past_kv_cache = [prompt_ids, kv_cache]

        if return_metrics:
            return input_ids, metrics
        return input_ids

    @staticmethod
    def reroot(tree, roots):
        """
        Discard nodes and fix references to previous nodes - keeping node order
        Returns pruned/merged tree and mask of kept nodes

        Input:
        ------
        tree - array[M, 2] - M node tree with [:, 1]-th child of parent [:, 0] at level [:, 2]
        """
        # todo better graph representation with fast prune/merge
        # old_id: (prev_old_id, child_id, level)
        new_nodes = {}
        counts = dict()
        # Merge root nodes
        tree[torch.isin(tree[:, 0], roots), 0] = roots[0]
        # Collect subtree
        queue = [(roots[0].item(), -1, -1)]
        while len(queue) > 0:
            cur, prev, prev_lvl = queue.pop(0)
            cid = counts.get(prev, -1) + 1
            counts[prev] = cid
            new_nodes[cur] = (prev, cid, prev_lvl + 1)
            for c in torch.where(tree[:, 0] == cur)[0]:
                queue += [(c.item(), cur, prev_lvl + 1)]
        idxs, nodes = zip(*sorted(new_nodes.items(), key=lambda x: x[0]))
        keep_mask = torch.isin(torch.arange(tree.shape[0]), torch.tensor(idxs)).to(device)
        # Re-index and remove root node
        keep_mask[0] = False
        new_tree = torch.tensor(nodes, device=device)[1:]
        new_tree[:, 0] = torch.tensor([idxs.index(i) for i in new_tree[:, 0]]) - 1
        new_tree[:, 2] -= 1
        return new_tree, keep_mask


    def proposal_trees(self, verify_tree=None, student_p=None, double_at=100, metrics=None):
        """
        Optimize proposals B wrt overhead adjusted expected # correct tokens
        max_B [sum_i A_i * H(B_i)] / [1 + sum_i B_i / 50)]

        If A_0 = 1 this becomes max_k H(k) / [1 + k / 50] = 14

        Input:
        ------
        verify_tree - array[M, 2] - M node tree with [:, 1]-th child of parent [:, 0] at level [:, 2]
        student_p   - array[M]    - student confidence for all nodes
        double_at   - slope of the transformers token vs time[s] curve, i.e. how many tokens needed to double the
                      inference time, i.e. each token adds 100/double_at % extra compute per call

        Output:
        -------
        props_tree - list[array[M_i, 2]], M + 1 trees with M_i nodes each
        """

        # Estimated probability of # correct tokens
        if student_p is None:
            A = self.hist_base
            if verify_tree is not None:
                n_verify = verify_tree[:, 2].max().item() if verify_tree.shape[0] > 0 else 0
                A = torch.cat([A[:n_verify], torch.tensor([A[n_verify:].sum()])])
        else:
            # assert student_p.shape[1] == n_verify
            # todo properly use verify_tree
            student_p = student_p[0:, :student_p.shape[1] // 2]
            A = torch.ones(student_p.shape[1] + 1)
            A[1:] = torch.cumprod(student_p[0].cpu(), dim=-1)
            A[:-1] *= 1 - student_p[0].cpu()
        # Reward; Estimated # correct tokens in the next step given k proposed tokens
        H_hist = torch.cumsum(self.hist_base * torch.arange(21, device=device), dim=-1)
        # or: # proposed tokens
        H_count = torch.arange(21, device=device)
        # switch dynamically
        if metrics is not None and len(metrics['Nrel']) > 0:
            rho = np.clip((np.array(metrics['Nrel']).mean() - 0.2) / 0.2, 0, 1)
        else:
            rho = 0
        rho = 0.5
        H = (1 - rho) * H_hist + rho * H_count

        # todo: add references for Dinkelbach?
        lam_pre, lam = 0, 0
        # todo _ = 3 to avoid check
        for _ in range(20):
            B = torch.argmax(A[:, None] * H[None, :self.tokens_per_student_call + 1] - lam * torch.arange(self.tokens_per_student_call + 1, device=A.device)[None, :] / double_at, dim=1)
            lam_pre = lam
            lam = torch.sum(A * H[B]) / (1 + B.sum() / double_at)
            if (lam_pre == lam).all():
                break

        props_tree = []
        assert B.sum() > 0
        for b in B:
            if b == 0:
                ths_tree = torch.zeros([0, 3], dtype=int, device=device)
            else:
                # Assuming line structure + 2-long branch
                # ths_tree = torch.empty([b+2, 3], dtype=int, device=device)
                # ths_tree[:b, 0] = torch.arange(-1, b - 1)
                # ths_tree[:b, 1] = 0
                # ths_tree[:b, 2] = torch.arange(b)
                # ths_tree[b:, 0] = -1
                # ths_tree[b:, 1] = torch.tensor([1, 2])
                # ths_tree[b:, 2] = torch.tensor([0, 0])
                # line + 1-long branches
                ths_tree = torch.empty([2 * b, 3], dtype=int, device=device)
                ths_tree[:b, 0] = torch.arange(-1, b - 1)
                ths_tree[:b, 1] = 0
                ths_tree[:b, 2] = torch.arange(b)
                ths_tree[b:, 0] = torch.arange(-1, b - 1)
                ths_tree[b:, 1] = 1
                ths_tree[b:, 2] = torch.arange(b)

            props_tree += [ths_tree]
        if student_p is not None:
            # props_tree += [props_tree[0].clone(), props_tree[0].clone()]

            for b in B[1:]:
                if b == 0:
                    ths_tree = torch.zeros([0, 3], dtype=int, device=device)
                else:
                    # Assuming line structure + 2-long branch
                    # ths_tree = torch.empty([b+2, 3], dtype=int, device=device)
                    # ths_tree[:b, 0] = torch.arange(-1, b - 1)
                    # ths_tree[:b, 1] = 0
                    # ths_tree[:b, 2] = torch.arange(b)
                    # ths_tree[b:, 0] = -1
                    # ths_tree[b:, 1] = torch.tensor([1, 2])
                    # ths_tree[b:, 2] = torch.tensor([0, 0])
                    # line + 1-long branches
                    ths_tree = torch.empty([2 * b, 3], dtype=int, device=device)
                    ths_tree[:b, 0] = torch.arange(-1, b - 1)
                    ths_tree[:b, 1] = 0
                    ths_tree[:b, 2] = torch.arange(b)
                    ths_tree[b:, 0] = torch.arange(-1, b - 1)
                    ths_tree[b:, 1] = 1
                    ths_tree[b:, 2] = torch.arange(b)

                props_tree += [ths_tree]

        assert len(props_tree) == verify_tree.shape[0] + 1
        return props_tree


        # # More guesses for speculated tokens
        #                 n_top_props = 1 * (n_props > 0).to(int)
        #                 ths_student_rank = student_logits[:,
        #                                    n_props[:num_correct].sum(): n_props[:num_correct].sum() + n_props[
        #                                        num_correct]].argsort(dim=-1, descending=True)
        #                 ths_student_predicted_more = torch.cat(
        #                     [ths_student_rank[:, d:d + 1, :n_top_prop] for d, n_top_prop in enumerate(n_top_props)])
        #
        #                 # ths_student_ps = torch.softmax(student_logits[:, n_props[:num_correct].sum()], dim=-1)
        #                 # sorted = ths_student_ps[0].argsort(descending=True)
        #                 # idx = torch.where(sorted == correct_tokens[0, num_correct])[0][0].item()
        #                 # metrics['off'] += [idx]
        #                 # metrics['offp'] += [ths_student_ps[0, sorted[idx]]]

    @torch.inference_mode()
    def generate_tree(self, batch, max_new_tokens, return_metrics=False, eos=None, force_correct_token=True, **kwargs):
        """
        Efficient Tree-based Coding using kv-cached Gated LoRA

        Parameters:
        -----------
        force_correct_token          - if True overwrite wrong tokens, assuming tokens predicted on the wrong token
                                       could still be correct. Causes higher #correct/call but smaller #calls/s
        self.tokens_per_student_call - max number of tokens to predict into the future, note that histogram-based
                                       proposal optimization in practice might impose an upper bound that is lower
        """
        # todo: replace fancy indexing with gather
        prompt_ids = batch['prompt_ids']
        assert prompt_ids.shape[0] == 1, "Batch size must be 1"
        assert self.student.shift_positions
        metrics = {'correct': [], 'N': [], 'Nrel': [], 'off': [], 'offp': []}
        timing = {'step': [], 'call': []}
        import time

        # Verify in parallel
        z_rnd_all = torch.rand([prompt_ids.shape[0], max_new_tokens + self.tokens_per_student_call], device=device, dtype=torch.float32)
        tokens_to_verify = max_new_tokens
        verify_tree = torch.zeros((0, 3), dtype=int, device=device)
        prop_trees = self.proposal_trees(verify_tree=verify_tree)

        # Fill caches
        # kv_cache = DynamicCache()
        scall = time.time()
        self.student.set_gate_window(0)
        outputs = self.student(
            input_ids=prompt_ids[:, :-1],
            use_cache=True,
            auxiliaries=z_rnd_all[:, :0],
        )
        kv_cache = outputs.past_key_values
        # torch.cuda.synchronize()
        timing['call'] += [time.time() - scall]

        while tokens_to_verify > 0:
            s = time.time()
            # Clip trees to not propose un-needed tokens, i.e. not more than max_new_tokens
            assert len(prop_trees) == verify_tree.shape[0] + 1
            for d, prop_tree in enumerate(prop_trees):
                verify_level = verify_tree[d - 1, 2] if d > 0 else -1
                prop_trees[d] = prop_tree[prop_tree[:, 2] <= tokens_to_verify - verify_level + 2]
            # todo remove leave nodes with child_id != 0
            n_props = [prop_tree.shape[0] for prop_tree in prop_trees]
            n_prop = sum(n_props)
            n_verify = verify_tree.shape[0]
            n_max_verify = verify_tree[:, 2].max().item() + 1 if n_verify > 0 else 0
            seq_len = prompt_ids.shape[1] + n_prop
            n_kv = kv_cache.get_seq_length()
            n_new = prompt_ids.shape[1] - n_kv
            n_old = n_new - n_verify
            n_check = n_verify + n_prop
            metrics['N'] += [n_new + n_prop]

            # Prepare inputs
            # todo: pre-allocate space for input mask tensor?
            z_idx = max_new_tokens - tokens_to_verify
            z_rnd = torch.cat([z_rnd_all[:, z_idx + prop_tree[:, 2] + (verify_tree[d - 1, 2] if d > 0 else -1)] for d, prop_tree in enumerate(prop_trees)], dim=1)
            input_ids = prompt_ids[:, kv_cache.get_seq_length():]
            input_mask = torch.tril(torch.ones(seq_len - kv_cache.get_seq_length(), seq_len, device=device), diagonal=kv_cache.get_seq_length())
            input_mask[n_old:, n_kv + n_old:] = 0
            input_mask.diagonal(offset=n_kv).fill_(1)
            for nd, node in enumerate(verify_tree[1:], start=1):
                input_mask[n_old + nd] = input_mask[n_old + max(0, node[0])]
                if node[0] == -1:
                    input_mask[n_old + nd, n_kv + n_old] = 0
                input_mask[n_old + nd, n_kv + n_old + nd] = 1
            input_position_ids = torch.arange(kv_cache.get_seq_length(), seq_len, device=device)
            input_position_ids[-n_check:-n_prop] = input_position_ids[-n_check-1] + verify_tree[:, 2] + 1
            midx = n_new
            for d, prop_tree in enumerate(prop_trees):
                verify_level = verify_tree[d - 1, 2] if d > 0 else -1
                # todo: is it faster to compute input_position_ids via level as for z_rnd?
                input_position_ids[midx: midx + prop_tree.shape[0]] = input_position_ids[-n_check-1] + prop_tree[:, 2] + verify_level + 1
                if prop_tree.shape[0] > 0:
                    input_mask[midx] = input_mask[n_old + max(0, d-1)]
                    if d == 0:
                        input_mask[midx, n_kv + n_old] = 0
                    input_mask[midx, n_kv + midx] = 1
                    for nd, node in enumerate(prop_tree[1:], start=1):
                        input_mask[midx + nd] = input_mask[midx + max(0, node[0])]
                        if node[0] == -1:
                            input_mask[midx + nd, n_kv + midx] = 0
                        input_mask[midx + nd, n_kv + midx + nd] = 1
                    midx += prop_tree.shape[0]
            # torch.set_printoptions(threshold=torch.inf, linewidth=1000); print(input_mask.to(int))
            input_mask = (1 - input_mask[None, None, :, :].to(torch.float16)) * -1e15
            input_position_ids = input_position_ids[None, :]

            # Student proposals
            scall = time.time()
            self.student.set_gate_window(n_prop)
            outputs = self.student(
                input_ids=input_ids,
                attention_mask=input_mask,
                position_ids=input_position_ids,
                auxiliaries=z_rnd,
                past_key_values=kv_cache,
                use_cache=True
            )
            # torch.cuda.synchronize()
            timing['call'] += [time.time() - scall]

            kv_cache = outputs.past_key_values
            full_logits = outputs.logits
            tgt_logits = full_logits[-1:, :n_new]
            # O-PTP
            assert 'bc' in self.loss_type
            student_logits = full_logits[:1, n_new:]
            student_p = torch.softmax(student_logits, dim=-1)
            idx_p_sorted = student_p.argsort(descending=True, dim=-1)
            ranks = torch.cat([prop_tree[:, 1] for prop_tree in prop_trees], dim=0)
            # student_predicted = idx_p_sorted[:, torch.arange(n_prop), ranks]
            student_predicted = idx_p_sorted.gather(dim=2, index=ranks[None, :, None]).squeeze(2)
            # student_predicted_max = student_logits.argmax(dim=-1)
            student_p_selected = student_p.gather(2, student_predicted[..., None]).squeeze(2)

            # Verify previously speculated tokens
            assert n_verify == tgt_logits.shape[1] - 1
            assert n_old == 1
            tgt_logits = self.adapt_logits(tgt_logits)
            right_bin_edges = torch.softmax(tgt_logits, dim=-1).cumsum(dim=-1)
            right_bin_edges[..., -1] = 1
            right_bin_edges[..., 0] = 0
            z_rnd_tgt = torch.cat([z_rnd_all[:, z_idx][:, None], z_rnd_all[:, z_idx + verify_tree[:, 2] + 1]], dim=1)
            correct_tokens = (right_bin_edges > z_rnd_tgt[..., None]).max(dim=-1).indices
            check_tokens = correct_tokens[:, verify_tree[:, 0] + 1]
            predict_tokens = prompt_ids[:, n_kv + n_old:]
            matches = (predict_tokens == check_tokens)
            # todo better graph representation with fast get_children
            verified = [-1]
            for _ in range(n_max_verify):
                children = torch.where(verify_tree[:, 0] == verified[-1])[0]
                matches_children = matches[:, children]
                if not matches_children.any():
                    break
                verified += [children[torch.where(matches_children[0])[0][0]].item()]
            verified = verified[1:]
            num_correct = len(verified)
            last_correct = verified[-1] if len(verified) > 0 else -1
            next_correct = correct_tokens[:, last_correct + 1]
            tokens_to_verify -= num_correct

            # Remove wrongly speculated tokens
            # kv_cache.crop(n_kv + n_old)
            for l in kv_cache.layers:
                l.keys = torch.cat([l.keys[..., :n_kv + n_old, :]] + [l.keys[..., n_kv + n_old + idx, :][..., None, :] for idx in verified], dim=-2)
                l.values = torch.cat([l.values[..., :n_kv + n_old, :]] + [l.values[..., n_kv + n_old + idx, :][..., None, :] for idx in verified], dim=-2)
            prompt_ids = torch.cat([prompt_ids[:, :n_kv + n_old]] + [prompt_ids[:, n_kv + n_old + idx][:, None] for idx in verified], dim=-1)
            if eos in prompt_ids[:, prompt_ids.shape[1] - num_correct:] or eos == next_correct:
                break

            # Update newly speculated tokens
            ths_prop_tree = prop_trees[last_correct + 1]
            skip_prop = sum(n_props[:last_correct + 1])
            ths_student_predicted = student_predicted[:, skip_prop: skip_prop + n_props[last_correct + 1]]
            ths_student_p = student_p[:, skip_prop: skip_prop + n_props[last_correct + 1]]
            ths_student_p_selected = student_p_selected[:, skip_prop: skip_prop + n_props[last_correct + 1]]
            if n_max_verify > 0:
                metrics['Nrel'] += [num_correct / n_max_verify]

            # Verify first speculated and add other speculated tokens
            new_roots = torch.where(ths_prop_tree[:, 0] == -1)[0]
            next_predicted = ths_student_predicted[:, new_roots]
            hits = torch.where(next_predicted == next_correct)[0]
            # todo pass full student_p instead and optimize full tree
            if len(hits) > 0:
                # Select only relevant subgraph
                ths_prop_tree, keep_mask = self.reroot(ths_prop_tree, new_roots[hits[0, None]])
                ths_student_predicted = ths_student_predicted[:, keep_mask]
                ths_student_p = ths_student_p_selected[:, keep_mask]
            elif force_correct_token and ths_prop_tree.shape[0] > 0:
                # Force correct tokens and reuse proposal
                ths_student_predicted[:, ths_prop_tree[:, 0] == last_correct] = next_correct
                ths_prop_tree, keep_mask = self.reroot(ths_prop_tree, new_roots)
                ths_student_predicted = ths_student_predicted[:, keep_mask]
                ths_student_p = ths_student_p_selected[:, keep_mask]
            else:
                # No proposal
                ths_prop_tree = torch.zeros([0, 3], dtype=int, device=device)
                ths_student_predicted = torch.zeros([1, 0], dtype=int, device=device)
                ths_student_p = None

            verify_tree = ths_prop_tree
            prop_trees = self.proposal_trees(verify_tree=ths_prop_tree, student_p=ths_student_p, metrics=metrics)
            prompt_ids = torch.cat([prompt_ids, next_correct[:, None], ths_student_predicted], dim=1)
            tokens_to_verify -= 1

            # torch.cuda.synchronize()
            timing['step'] += [time.time() - s]
            metrics['correct'] += [num_correct + 1]

        print('-', np.mean(metrics['N']))
        metrics = {
            'completion': prompt_ids,
            'correct_per_call': np.mean(metrics['correct']),
            'correct_all': metrics['correct'],
            'num_calls': len(metrics['correct']),
        }

        if return_metrics:
            return input_ids, metrics
        return input_ids
