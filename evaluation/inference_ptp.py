"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse

from evaluation.eval import run_eval, reorg_answer_file

from fastchat.utils import str_to_torch_dtype

from transformers import AutoModelForCausalLM, AutoTokenizer
from model.ptp.lit import ParallelSamplingLightningModule
from model.ptp.utils import instantiate
import os
from omegaconf import DictConfig, OmegaConf
import yaml
import lightning.pytorch
import torch

def ptp_forward(inputs, model, tokenizer, max_new_tokens, do_sample=True, temperature=0.0):
    model.temperature = temperature
    if not list(model.teacher.parameters())[0].device.type == 'cuda':
        model.to('cuda')

    metrics = model.timed_error_correction({'prompt_ids': inputs.input_ids}, tokens_to_fill=max_new_tokens, eos=tokenizer.eos_token_id)
    output_ids = metrics['completion']
    step = metrics['num_calls']
    accept_length_list = metrics['all_correct']
    new_token = sum(accept_length_list)

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
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    with open(os.path.join(args.student_path, 'validate.yaml'), 'r') as f:
        config = DictConfig(yaml.safe_load(f))
    torch.set_float32_matmul_precision('medium')

    lit_model = instantiate(config['model'])
    ckpt = torch.load(os.path.join(args.student_path, 'last-farrin.ckpt'), map_location='cuda')
    lit_model.load_state_dict(ckpt['state_dict'], strict=False)

    tokenizer = lit_model.teacher.tokenizer

    lit_model.teacher.eval()
    lit_model.student.eval()

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