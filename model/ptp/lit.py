import time
from lightning.pytorch import LightningModule
from typing import Literal, List, Mapping, Any
from model.ptp.transformer import TransformerModel, MixedTransformerModel
import random
import torch
import numpy as np
from transformers.cache_utils import DynamicCache


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

    # @torch.inference_mode()
    # def generate(self, batch, max_new_tokens=None, max_length=None, attention_mask=None,
    #              use_cache=True, return_metrics=False, do_sample=True, stopping_criteria=None,
    #              pad_token_id=None, error_correction=False, eos=None, recompute=True):
    #     input_ids = batch['prompt_ids']
    #     assert input_ids.shape[0] == 1, "Batch size must be 1 for now"
    #
    #     if max_length is not None:
    #         assert max_new_tokens is None, "Only one of max_length and max_new_tokens can be set"
    #         max_new_tokens = max_length - input_ids.shape[1]
    #     assert max_new_tokens is not None, "One of max_length and max_new_tokens must be set"
    #     assert max_new_tokens > 0, f"max_new_tokens must be positive, got {max_new_tokens}"
    #     if not do_sample:
    #         import warnings
    #         warnings.warn("do_sample=False is not supported, proceeding with sampling anyway.")
    #
    #     # This would be faster using completion_ids and bin_edges but we want to measure the verification call.
    #     if error_correction:
    #         self.teacher.eval()
    #     prompt_ids = input_ids
    #     # prompt_ids = prompt_ids.repeat(20, 1)
    #     # if attention_mask is None:
    #     #     attention_mask = torch.ones_like(prompt_ids, dtype=torch.long, device=prompt_ids.device)
    #     # prompt_mask = attention_mask
    #     device = prompt_ids.device
    #     kv_student = None
    #     kv_teacher = None
    #
    #     metrics = {
    #         'accuracy': [],
    #         'correct': [],
    #         'time_student': [],
    #         'time_teacher': [],
    #         'total_time': [],
    #         'token_per_time': []
    #     }
    #     # if all predicted tokens are correct, verification can add one extra token
    #     tokens_to_fill = max_new_tokens
    #     z_rnd_all = torch.rand([prompt_ids.shape[0], max_new_tokens + 1], device=device, dtype=torch.float32)
    #     if not recompute:
    #         left_bin_edges = batch['left_bin_edges']
    #         right_bin_edges = batch['right_bin_edges']
    #         z_rnd_all[:, :left_bin_edges.shape[1]] = left_bin_edges + (right_bin_edges - left_bin_edges) * z_rnd_all[:, :left_bin_edges.shape[1]]
    #
    #     while tokens_to_fill > 0:
    #         # Student proposal
    #         n_prop = min(self.student_calls_per_step * self.tokens_per_student_call, tokens_to_fill)
    #         z_idx = max_new_tokens - tokens_to_fill
    #         z_rnd = z_rnd_all[:, z_idx:z_idx + n_prop + 1]
    #         start_time = time.time()
    #         for k in range(min(self.student_calls_per_step, np.ceil(tokens_to_fill / self.tokens_per_student_call))):
    #             ths_n_prop = min((k + 1) * self.tokens_per_student_call,
    #                              tokens_to_fill) - k * self.tokens_per_student_call
    #             # input_mask = torch.cat([
    #             #     prompt_mask,
    #             #     torch.ones([prompt_ids.shape[0], ths_n_prop], device=prompt_mask.device, dtype=prompt_mask.dtype)
    #             # ], dim=1)
    #             if use_cache:
    #                 outputs = self.student(
    #                     input_ids=prompt_ids if kv_student is None else prompt_ids[:, kv_student.get_seq_length():],
    #                     # attention_mask=input_mask,
    #                     auxiliaries=z_rnd[
    #                         :, k * self.tokens_per_student_call:k * self.tokens_per_student_call + ths_n_prop],
    #                     past_key_values=kv_student,
    #                     use_cache=True
    #                 )
    #                 full_logits = outputs.logits
    #                 kv_student = outputs.past_key_values
    #             else:
    #                 full_logits = self.student(
    #                     input_ids=prompt_ids,
    #                     # attention_mask=input_mask,
    #                     auxiliaries=z_rnd[
    #                         :, k * self.tokens_per_student_call:k * self.tokens_per_student_call + ths_n_prop],
    #                 ).logits
    #             if self.loss_type.startswith('kl'):
    #                 raise NotImplementedError
    #             elif 'bc' in self.loss_type:
    #                 # ------
    #                 # O-PTP
    #                 # ------
    #                 student_logits = full_logits[:, -ths_n_prop:]
    #                 student_predicted = student_logits.argmax(dim=2)
    #                 student_completion = torch.cat([
    #                     prompt_ids,
    #                     student_predicted,
    #                 ], dim=1)
    #             else:
    #                 raise NotImplementedError(self.loss_type)
    #             prompt_ids = student_completion
    #             # prompt_mask = input_mask
    #         metrics['time_student'] += [time.time() - start_time]
    #
    #         # Teacher verification
    #         start_time = time.time()
    #         if use_cache:
    #             # if kv_teacher is None and kv_student is not None:
    #             #     from copy import deepcopy
    #             #     kv_teacher = deepcopy(kv_student)
    #             #     kv_teacher.crop(input_ids.shape[1])
    #             outputs = self.teacher(
    #                 input_ids=prompt_ids if kv_teacher is None else prompt_ids[:, kv_teacher.get_seq_length():],
    #                 # attention_mask=prompt_mask,
    #                 past_key_values=kv_teacher,
    #                 use_cache=True
    #             )
    #             kv_teacher = outputs.past_key_values
    #         else:
    #             outputs = self.teacher(
    #                 input_ids=prompt_ids,
    #                 # attention_mask=prompt_mask,
    #                 return_scores=True
    #             )
    #
    #         tgt_logits = outputs.logits[..., - n_prop - 1:, :]
    #         tgt_logits = self.adapt_logits(tgt_logits)
    #         # ? assert logits.dtype == torch.float32, logits.dtype
    #         right_bin_edges = torch.softmax(tgt_logits, dim=-1).cumsum(dim=-1)
    #         right_bin_edges[..., -1] = 1
    #         right_bin_edges[..., 0] = 0
    #         correct_tokens = (right_bin_edges > z_rnd[..., None]).max(dim=-1).indices
    #         # print(correct_tokens.min(), correct_tokens.max())
    #         # print('a')
    #         # correct_right = torch.gather(right_bin_edges, dim=2, index=correct_tokens[..., None])[..., 0]
    #         # correct_left = torch.gather(right_bin_edges, dim=2, index=correct_tokens[..., None]-1)[..., 0]
    #         # print('b')
    #         # pos_in_interval = (z_rnd - correct_left) / (correct_right - correct_left)
    #         if not recompute:
    #             correct_tokens = batch['completion_ids'][:, z_idx:z_idx + n_prop]
    #         if self.student_calls_per_step >= 1:
    #             if not recompute:
    #                 check_tokens = correct_tokens
    #             else:
    #                 check_tokens = correct_tokens[..., :-1]
    #             # predict_tokens = prompt_ids[..., - n_prop:]
    #             predict_tokens = prompt_ids[..., - n_prop:][..., :correct_tokens.shape[1]]
    #             num_correct = (check_tokens == predict_tokens).float().argmin(dim=1)
    #             num_correct[(check_tokens == predict_tokens).all(dim=1)] = n_prop
    #
    #             idx = 0 # int(torch.abs(pos_in_interval - 0.5)[:, 0].argmin())
    #             num_correct = min(int(num_correct[idx].float().mean()), 15)
    #             accuracy = (check_tokens == predict_tokens)[idx].float().mean().item()
    #         else:
    #             num_correct = 0
    #             accuracy = 1.0
    #         metrics['accuracy'] += [accuracy]
    #         metrics['correct'] += [num_correct + 1]
    #
    #         # Fix first mistake, only works for batch_size=1 for now
    #         pos = int(prompt_ids.shape[-1] - (n_prop - num_correct))
    #         pos_pre = int(prompt_ids.shape[-1] - n_prop)
    #         if n_prop > num_correct:
    #             prompt_ids = prompt_ids[idx, :pos].expand(prompt_ids.shape[0], -1)
    #             # prompt_mask = prompt_mask[..., :pos]
    #         # prune kv-cache
    #         if kv_teacher is not None:
    #             kv_teacher.crop(pos)
    #             # for l in kv_teacher.layers:
    #             #     l.keys = l.keys[idx, ...].expand(prompt_ids.shape[0], *[-1] * 3)
    #             #     l.values = l.values[idx, ...].expand(prompt_ids.shape[0], *[-1] * 3)
    #         if kv_student is not None:
    #             kv_student.crop(pos_pre)
    #             # for l in kv_student.layers:
    #             #     l.keys = l.keys[idx, ...].expand(prompt_ids.shape[0], *[-1] * 3)
    #             #     l.values = l.values[idx, ...].expand(prompt_ids.shape[0], *[-1] * 3)
    #         prompt_ids = torch.cat([
    #             prompt_ids,
    #             correct_tokens[idx:idx+1, num_correct][:, None].repeat(prompt_ids.shape[0], 1),
    #         ], dim=1)
    #         # prompt_mask = torch.cat([
    #         #     prompt_mask,
    #         #     torch.ones([1, 1], device=prompt_mask.device, dtype=prompt_mask.dtype),
    #         # ], dim=1)
    #         tokens_to_fill -= (num_correct + 1)
    #
    #         metrics['time_teacher'] += [time.time() - start_time]
    #
    #         if eos in prompt_ids[idx, pos_pre:]:
    #             break
    #         # if stopping_criteria is not None and stopping_criteria(prompt_ids):
    #         #     break
    #
    #     total_time = sum(metrics['time_student'] + metrics['time_teacher'])
    #     metrics = {
    #         'completion': prompt_ids,
    #         'accuracy': np.mean(metrics['accuracy']),
    #         'correct_per_call': np.mean(metrics['correct']) - 1,
    #         'correct_first': metrics['correct'][0] - 1,
    #         'correct_second': metrics['correct'][1] - 1,
    #         'correct_all': metrics['correct'],
    #         'time_student': np.mean(metrics['time_student']),
    #         'time_teacher': np.mean(metrics['time_teacher']),
    #         'time': total_time,
    #         'num_calls': len(metrics['correct']),
    #         'token_per_time': (sum(metrics['correct'])) / total_time
    #     }
    #
    #     if return_metrics:
    #         return input_ids, metrics
    #     return input_ids

    @torch.inference_mode()
    def generate(self, batch, max_new_tokens=None, max_length=None, attention_mask=None,
                 use_cache=True, return_metrics=False, do_sample=True, stopping_criteria=None,
                 pad_token_id=None, error_correction=False, eos=None, recompute=True):
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
        prompt_ids = input_ids
        prompt_ids = prompt_ids.repeat(2, 1)
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

        while tokens_to_fill > 0:
            # Student proposal
            n_prop = min(self.tokens_per_student_call, tokens_to_fill)
            z_idx = max_new_tokens - tokens_to_fill
            z_rnd = z_rnd_all[:, z_idx:z_idx + n_prop + 1]
            start_time = time.time()
            # input_mask = torch.cat([
            #     prompt_mask,
            #     torch.ones([prompt_ids.shape[0], ths_n_prop], device=prompt_mask.device, dtype=prompt_mask.dtype)
            # ], dim=1)
            if use_cache:
                outputs = self.student(
                    input_ids=prompt_ids if kv_student is None else prompt_ids[:, kv_student.get_seq_length():],
                    # attention_mask=input_mask,
                    auxiliaries=z_rnd[:, :-1],
                    past_key_values=kv_student,
                    use_cache=True
                )
                full_logits = outputs.logits
                kv_student = outputs.past_key_values
            else:
                full_logits = self.student(
                    input_ids=prompt_ids,
                    # attention_mask=input_mask,
                    auxiliaries=z_rnd[:, :-1],
                ).logits
            if self.loss_type.startswith('kl'):
                raise NotImplementedError
            elif 'bc' in self.loss_type:
                # ------
                # O-PTP
                # ------
                student_logits = full_logits[:1, -n_prop:]
                student_predicted = student_logits.argmax(dim=2)
                student_completion = torch.cat([
                    prompt_ids[1:],
                    student_predicted,
                ], dim=1)
            else:
                raise NotImplementedError(self.loss_type)
            prompt_ids = student_completion.repeat(2, 1)
            # prompt_mask = input_mask
            metrics['time_student'] += [time.time() - start_time]

            # Teacher verification
            start_time = time.time()
            if use_cache:
                outputs = super(MixedTransformerModel, self.student).forward(
                    input_ids=prompt_ids if kv_teacher is None else prompt_ids[:, kv_teacher.get_seq_length():],
                    # attention_mask=prompt_mask,
                    past_key_values=kv_teacher,
                    use_cache=True
                )
                kv_teacher = outputs.past_key_values
            else:
                outputs = super(MixedTransformerModel, self.student).forward(
                    input_ids=prompt_ids,
                    # attention_mask=prompt_mask,
                    return_scores=True
                )

            tgt_logits = outputs.logits[-1:, - n_prop - 1:, :]
            tgt_logits = self.adapt_logits(tgt_logits)
            # ? assert logits.dtype == torch.float32, logits.dtype
            right_bin_edges = torch.softmax(tgt_logits, dim=-1).cumsum(dim=-1)
            right_bin_edges[..., -1] = 1
            right_bin_edges[..., 0] = 0
            correct_tokens = (right_bin_edges > z_rnd[..., None]).max(dim=-1).indices
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

                idx = 0  # int(torch.abs(pos_in_interval - 0.5)[:, 0].argmin())
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
            if kv_student is not None:
                kv_student.crop(pos_pre)
                # for l in kv_student.layers:
                #     l.keys = l.keys[idx, ...].expand(prompt_ids.shape[0], *[-1] * 3)
                #     l.values = l.values[idx, ...].expand(prompt_ids.shape[0], *[-1] * 3)
            prompt_ids = torch.cat([
                prompt_ids,
                correct_tokens[idx:idx + 1, num_correct][:, None].repeat(prompt_ids.shape[0], 1),
            ], dim=1)
            # prompt_mask = torch.cat([
            #     prompt_mask,
            #     torch.ones([1, 1], device=prompt_mask.device, dtype=prompt_mask.dtype),
            # ], dim=1)
            tokens_to_fill -= (num_correct + 1)

            metrics['time_teacher'] += [time.time() - start_time]

            if eos in prompt_ids[idx, pos_pre:]:
                break
            # if stopping_criteria is not None and stopping_criteria(prompt_ids):
            #     break

        total_time = sum(metrics['time_student'] + metrics['time_teacher'])
        metrics = {
            'completion': prompt_ids,
            'accuracy': np.mean(metrics['accuracy']),
            'correct_per_call': np.mean(metrics['correct']) - 1,
            'correct_first': metrics['correct'][0] - 1,
            'correct_second': metrics['correct'][1] - 1,
            'correct_all': metrics['correct'],
            'time_student': np.mean(metrics['time_student']),
            'time_teacher': np.mean(metrics['time_teacher']),
            'time': total_time,
            'num_calls': len(metrics['correct']),
            'token_per_time': (sum(metrics['correct'])) / total_time
        }

        if return_metrics:
            return input_ids, metrics
        return input_ids
