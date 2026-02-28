"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse

import torch.cuda
from fastchat.utils import str_to_torch_dtype
from pandas.core.dtypes.inference import is_number
from scipy.signal import max_len_seq

from evaluation.eval import run_eval, reorg_answer_file

from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
from transformers.cache_utils import DynamicCache
import math


def baseline_forward(inputs, model, tokenizer, max_new_tokens, temperature=0.0, do_sample=False):
    input_ids = inputs.input_ids
    import time
    s = time.time()

    # output_ids = model.generate(
    #     input_ids,
    #     do_sample=do_sample,
    #     temperature=temperature,
    #     max_new_tokens=max_new_tokens,
    # )

    timing = {'call': []}
    kv_cache = StaticCache(model.config, batch_size=1, max_cache_len=2048)
    generated = input_ids
    input_ids = generated[:, :-1]
    raw_len = input_ids.shape[1]
    padded_len = 1 << ((raw_len - 1).bit_length() if raw_len > 1 else 0)
    if padded_len > raw_len:
        input_ids = torch.cat([input_ids, input_ids.new_full((1, padded_len - raw_len), 13)], dim=1)
    outputs = model(
        input_ids=input_ids,
        past_key_values=kv_cache,
        use_cache=True
    )
    kv_cache.crop(generated.shape[1] - 1)
    kv_cache = outputs.past_key_values
    for step in range(max_new_tokens):
        scall = time.time()
        torch.compiler.cudagraph_mark_step_begin()
        outputs = model(
            input_ids=generated[:, -1:],
            past_key_values=kv_cache,
            use_cache=True,
        )
        kv_cache = outputs.past_key_values
        timing['call'] += [time.time() - scall]
        logits = outputs.logits[:, -1, :]
        if temperature > 0.0:
            next_token = torch.distributions.Categorical(logits=logits / temperature).sample()[None, :]
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)
        if next_token.item() == tokenizer.eos_token_id:
           break
    output_ids = generated

    torch.cuda.synchronize()
    timed = time.time() - s
    new_token = len(output_ids[0][len(input_ids[0]):])
    print(timed, new_token/timed, 1000 * timed/new_token)
    step = new_token
    accept_length_list = [1] * new_token
    return output_ids, new_token, step, accept_length_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
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
        default=0.0,
        help="The temperature for medusa sampling.",
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

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=str_to_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
    )
    torch.set_float32_matmul_precision('high')
    model.to('cuda')
    model.eval()

    # torch._dynamo.config.suppress_errors = False
    # torch._dynamo.config.verbose = True
    # model.generation_config.cache_implementation = "static"
    # model = torch.compile(model, mode='reduce-overhead', dynamic=False, fullgraph=True)
    model = torch.compile(model, mode='default', dynamic=True, fullgraph=False)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)

    if args.temperature > 0:
        do_sample = True
    else:
        do_sample = False

    run_eval(
        model=model,
        tokenizer=tokenizer,
        forward_func=baseline_forward,
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
        seed_shift = 1000 * int(args.model_id[-1]) if args.model_id[-1].isdigit() else 0,
    )

    reorg_answer_file(answer_file)