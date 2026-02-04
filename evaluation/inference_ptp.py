"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse

from evaluation.eval import run_eval, reorg_answer_file
from fastchat.utils import str_to_torch_dtype

from model.ptp.utils import instantiate
import os
from omegaconf import DictConfig
import yaml
import torch

STPS = []

def ptp_forward(inputs, model, tokenizer, max_new_tokens, do_sample=True, temperature=0.0, prepare=False):
    # if not list(model.teacher.parameters())[0].device.type == 'cuda':
    #     model.to('cuda')
    if prepare:
        return
        # outputs = model.teacher.generate(
        #     input_ids=inputs.input_ids,
        #     max_new_tokens=max_new_tokens,
        #     do_sample=True,
        #     # pad_token_id=tokenizer.eos_token_id,
        #     temperature=temperature,
        #     top_p=model.top_p,
        #     top_k=model.top_k,
        #     output_scores=True,
        #     return_dict_in_generate=True,
        #     num_return_sequences=1,
        #     use_cache=True,
        # )
        # chunk_ids = outputs.sequences[:, inputs.input_ids.shape[1]:]
        # scores = torch.stack(outputs.scores, dim=1)
        # probs = torch.softmax(scores, dim=-1)
        # selector = (
        #     torch.arange(probs.shape[0], device=probs.device)[:, None],
        #     torch.arange(probs.shape[1], device=probs.device)[None, :],
        #     chunk_ids
        # )
        # assert (probs[selector] > 0).all(), "Some token has zero probability"
        # cum_probs = probs.cumsum(-1)
        #
        # model.prep_completion_ids = chunk_ids
        # model.prep_left_bin_edges = cum_probs[selector] - probs[selector]
        # model.prep_right_bin_edges = cum_probs[selector]
        # return
    else:
        model.temperature = temperature

        # input_ids = torch.tensor(tokenizer.encode("A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nUSER: How can cross training benefit groups like runners, swimmers, or weightlifters?\nASSISTANT: "))[None, :].to('cuda')
        import time
        s = time.time()
        # metrics = model.timed_error_correction({'prompt_ids': input_ids}, tokens_to_fill=max_new_tokens, eos=tokenizer.eos_token_id)

        # model.student._gated_window = 16
        # outputs = model.student(
        #     input_ids=inputs['input_ids'][:, :10],
        #     auxiliaries=torch.rand([1, 16], device='cuda', dtype=torch.float32)
        # )
        # import time
        # s = time.time()
        # past_key_values = None
        # generated = inputs['input_ids']
        # n_ths = 1
        # model.student._gated_window = n_ths
        # model.student._mode = 'student'
        # for step in range(1024):
        #     outputs = model.student(
        #         input_ids=generated[:, -1:] if past_key_values is not None else generated,
        #         past_key_values=past_key_values,
        #         use_cache=True,
        #         auxiliaries=torch.rand([1, n_ths], device='cuda', dtype=torch.float32)
        #     )
        #     logits = outputs.logits[:, -n_ths-1, :]
        #     past_key_values = outputs.past_key_values
        #     probs = torch.softmax(logits, dim=-1)
        #     next_token = torch.multinomial(probs, num_samples=1)
        #     # next_token = torch.argmax(logits, dim=-1, keepdim=True)
        #     past_key_values.crop(generated.shape[1])
        #     generated = torch.cat([generated, next_token], dim=-1)
        #     # if next_token.item() == tokenizer.eos_token_id:
        #     #     break
        # output_ids = generated
        # torch.cuda.synchronize()
        # timed = time.time() - s
        # new_token = len(output_ids[0][len(inputs['input_ids'][0]):])
        # print(timed, new_token / timed, 1000 * timed / new_token)

        metrics = model.generate(
            {'prompt_ids': inputs.input_ids,
             # 'completion_ids': model.prep_completion_ids,'left_bin_edges':model.prep_left_bin_edges, 'right_bin_edges':model.prep_right_bin_edges
             },
            max_new_tokens=max_new_tokens, return_metrics=True, eos=tokenizer.eos_token_id, gated=True, correcting_via='u', collect_stats=True)[1]
        output_ids = metrics['completion']
        step = metrics['num_calls']
        accept_length_list = metrics['correct_all']
        new_token = sum(accept_length_list)

        # global STPS
        # STPS += [metrics['STP']]

        # model.prep_completion_ids = None
        # model.prep_left_bin_edges = None
        # model.prep_right_bin_edges = None

        # torch.cuda.synchronize()
        timed = time.time() - s
        print(timed, new_token / timed, 1000 * timed / step)
    return output_ids, new_token, step, accept_length_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-model-path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--student-path",
        type=str,
        required=True,
    )
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end",
        type=int,
        help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="The temperature for ptp sampling.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float64", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU.",
    )
    args = parser.parse_args()

    question_file = f"data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}-float16-temp-0.7-lorag128_paper_fast.jsonl"

    print(f"Output to {answer_file}")

    with open(os.path.join(args.student_path, 'validate_spec.yaml'), 'r') as f:
        config = DictConfig(yaml.safe_load(f))
    torch.set_float32_matmul_precision('medium')

    lit_model = instantiate(config['model'])
    # ckpt = torch.load(os.path.join(args.student_path, 'last-farrin.ckpt'), map_location='cuda')
    # ckpt = torch.load(os.path.join(args.student_path, 'vicuna-7b-ultrachat-lora-r64/epoch=86.ckpt'), map_location='cuda')
    # ckpt = torch.load(os.path.join(args.student_path, 'vicuna-7b-ultrachat-lora-r8/gated/epoch=9.ckpt'), map_location='cuda')
    # ckpt = torch.load(os.path.join(args.student_path, 'vicuna-7b-ultrachat-lora-r128/gated/epoch=424.ckpt'), map_location='cuda')
    # ckpt = torch.load(os.path.join(args.student_path, 'vicuna-7b-ultrachat-lora-r128/epoch=484.ckpt'), map_location='cuda')
    ckpt = torch.load(os.path.join(args.student_path, 'vicuna-7b-sharegpt-lora-r128/gated/epoch=140.ckpt'), map_location='cuda')
    # mistakes = lit_model.load_state_dict({k.replace('model.base_model.', '').replace('base_layer.', ''): v for k, v in ckpt['state_dict'].items()}, strict=False)
    # mistakes = lit_model.load_state_dict({k.replace('base_layer.', ''): v for k, v in ckpt['state_dict'].items()}, strict=False)
    mistakes = lit_model.load_state_dict(ckpt['state_dict'], strict=False)
    assert all(key.startswith('teacher') or 'lora_linear' in key for key in mistakes.missing_keys)

    tokenizer = lit_model.student.tokenizer

    # lit_model.teacher.eval()
    lit_model.student.eval()
    lit_model.student.model.merge_and_unload(progressbar=True)
    # lit_model.teacher.compile()
    # lit_model.student.compile()
    lit_model.to(str_to_torch_dtype(args.dtype))
    lit_model.to('cuda')

    if args.temperature > 0:
        do_sample = True
    else:
        do_sample = False

    run_eval(
        model=lit_model,
        tokenizer=tokenizer,
        forward_func=ptp_forward,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_tokens=args.max_new_tokens,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        temperature=args.temperature,
        do_sample=do_sample,
    )

    reorg_answer_file(answer_file)