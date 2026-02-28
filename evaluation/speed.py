import json
import argparse
from transformers import AutoTokenizer
import numpy as np
import re

def get_last_number(text):
    matches = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', text)
    if not matches:
        return np.nan
    num = matches[-1].replace(',', '')
    if '.' in num:
        return float(num)
    return int(num)

def correctness(jsonl_file, jsonl_file_base, tokenizer_path, report=True):
    task = 'math_reasoning'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    from fastchat.llm_judge.common import load_questions
    questions = load_questions('../data/spec_bench/question.jsonl', None, None)
    all_correct_answers = {401: 1000, 402: 5, 403: 306, 404: 7, 405: 3, 406: 2, 407: 6, 408: 1000, 409: 26, 410: 72, 411: 22, 412: 25, 413: 110, 414: 10, 415: 5, 416: 2, 417: 5, 418: -3, 419: 240, 420: 320, 421: 4, 422: 284, 423: 7, 424: 1300, 425: 30, 426: 5, 427: 19, 428: 95200, 429: 385000, 430: 84, 431: 15, 432: 34, 433: 84, 434: 360, 435: 132, 436: 4, 437: 36, 438: 8, 439: 1110, 440: 120, 441: 132, 442: 10, 443: 54, 444: 3, 445: 2, 446: 40, 447: 4, 448: 40, 449: 623, 450: 9, 451: 18, 452: 36, 453: 35, 454: 80, 455: 120, 456: 40, 457: 15, 458: 21, 459: 120, 460: 20, 461: 6, 462: 40, 463: 5, 464: 336, 465: 40, 466: 350, 467: 7, 468: 5, 469: 100, 470: 105, 471: 16, 472: 8, 473: 250, 474: 48, 475: 5760, 476: 13, 477: 360, 478: 36, 479: 762, 480: 576}

    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            if json_obj["category"] == task:
                data.append(json_obj)

    student_answers = []
    for datapoint in data:
        text = datapoint["choices"][0]['turns'][0]
        if datapoint['question_id'] not in all_correct_answers:
            for q in questions:
                if q['question_id'] == datapoint['question_id']:
                    print(q['question_id'])
                    print(q['turns'][0])
                    all_correct_answers[q['question_id']] = int(input())
        answer = get_last_number(text)
        student_answers += [answer]

    data = []
    with open(jsonl_file_base, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            if json_obj["category"] == task:
                data.append(json_obj)

    base_answers = []
    correct_answers = []
    for datapoint in data:
        text = datapoint["choices"][0]['turns'][0]
        answer = get_last_number(text)
        base_answers += [answer]
        correct_answers += [all_correct_answers[datapoint['question_id']]]

    student_answers = np.array(student_answers)
    base_answers = np.array(base_answers)
    correct_answers = np.array(correct_answers)
    if report:
        print("="*30, "Task: ", task, "="*30)
        print("Answer Correct: %.2f %%"  % (100 * np.mean(student_answers == correct_answers)))
        print("Answer Match  : %.2f %%" % (100 * np.mean(student_answers == base_answers)))
        print("Base   Correct: %.2f %%" % (100 * np.mean(correct_answers == base_answers)))
    return


def speed(jsonl_file, jsonl_file_base, tokenizer, task=None, report=True):
    tokenizer=AutoTokenizer.from_pretrained(tokenizer)
    mt_bench_list = ["writing", "roleplay", "reasoning", "math" , "coding", "extraction", "stem", "humanities"]

    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            if task=="overall":
                data.append(json_obj)
            elif task == "mt_bench":
                if json_obj["category"] in mt_bench_list:
                    data.append(json_obj)
            else:
                if json_obj["category"] == task:
                    data.append(json_obj)

    speeds=[]
    accept_lengths_list = []
    for datapoint in data:
        tokens=sum(datapoint["choices"][0]['new_tokens'])
        times = sum(datapoint["choices"][0]['wall_time'])
        accept_lengths_list.extend(datapoint["choices"][0]['accept_lengths'])
        speeds.append(tokens/times)


    data = []
    with open(jsonl_file_base, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            if task=="overall":
                data.append(json_obj)
            elif task == "mt_bench":
                if json_obj["category"] in mt_bench_list:
                    data.append(json_obj)
            else:
                if json_obj["category"] == task:
                    data.append(json_obj)

    total_time=0
    total_token=0
    speeds0=[]
    for datapoint in data:
        answer=datapoint["choices"][0]['turns']
        tokens = 0
        for i in answer:
            tokens += (len(tokenizer(i).input_ids) - 1)
        times = sum(datapoint["choices"][0]['wall_time'])
        speeds0.append(tokens / times)
        total_time+=times
        total_token+=tokens

    tokens_per_second = np.array(speeds).mean()
    tokens_per_second_baseline = np.array(speeds0).mean()
    speedup_ratio = np.array(speeds).mean()/np.array(speeds0).mean()

    if report:
        print("="*30, "Task: ", task, "="*30)
        print("#Mean accepted tokens: ", np.mean(accept_lengths_list))
        print('Tokens per second: ', tokens_per_second)
        print('Tokens per second for the baseline: ', tokens_per_second_baseline)
        print("Speedup ratio: ", speedup_ratio)
    return tokens_per_second, tokens_per_second_baseline, speedup_ratio, accept_lengths_list


def get_single_speedup(jsonl_file, jsonl_file_base, tokenizer_path):
    for subtask_name in ["mt_bench", "translation", "summarization", "qa", "math_reasoning", "rag", "overall"]:
        speed(jsonl_file, jsonl_file_base, tokenizer_path, task=subtask_name)


def get_mean_speedup(jsonl_file, jsonl_file_base, tokenizer_path):
    jsonl_file_run_list = ['%s-%d.jsonl' % (jsonl_file[:-6], i) for i in range(1, 4)]
    jsonl_file_base_run_list = ['%s-%d.jsonl' % (jsonl_file_base[:-6], i) for i in range(1, 4)]

    for subtask_name in ["mt_bench", "translation", "summarization", "qa", "math_reasoning", "rag", "overall"]:
        print("=" * 30, "Task: ", subtask_name, "=" * 30)
        tokens_per_second_list = []
        tokens_per_second_baseline_list = []
        speedup_ratio_list = []
        accept_lengths_list = []
        for jsonl_file, jsonl_file_base in zip(jsonl_file_run_list, jsonl_file_base_run_list):
            tokens_per_second, tokens_per_second_baseline, speedup_ratio, accept_lengths = speed(jsonl_file, jsonl_file_base, tokenizer_path, task=subtask_name, report=False)
            tokens_per_second_list.append(tokens_per_second)
            tokens_per_second_baseline_list.append(tokens_per_second_baseline)
            speedup_ratio_list.append(speedup_ratio)
            accept_lengths_list.extend(accept_lengths)
            print('%.4f' % np.mean(accept_lengths))

        avg_accept_lengths = np.mean(accept_lengths_list)
        se_accept_lengths = np.std(accept_lengths_list, ddof=1) / np.sqrt(len(accept_lengths_list))
        # print("#Mean accepted tokens: {}, SE result: {}".format(avg_accept_lengths, se_accept_lengths))
        print("#Mean accepted tokens: %.2f +- %.4f" % (avg_accept_lengths, se_accept_lengths))

        avg = np.mean(tokens_per_second_list)
        std = np.std(tokens_per_second_list, ddof=1)  # np.sqrt(( a.var() * a.size) / (a.size - 1))
        print("Tokens per second: Mean result: {}, Std result: {}".format(avg, std))

        avg_baseline = np.mean(tokens_per_second_baseline_list)
        std_baseline = np.std(tokens_per_second_baseline_list, ddof=1)  # np.sqrt(( a.var() * a.size) / (a.size - 1))
        print("Tokens per second (baseline): Mean result: {}, Std result: {}".format(avg_baseline, std_baseline))

        avg_speedup = np.mean(speedup_ratio_list)
        std_speedup = np.std(speedup_ratio_list, ddof=1)  # np.sqrt(( a.var() * a.size) / (a.size - 1))
        print("Speedup ratio: Mean result: {}, SE result: {}".format(avg_speedup, std_speedup  / np.sqrt(3)))
        print("Speedup ratio: %.2f +- %.4f" % (avg_speedup, std_speedup / np.sqrt(3)))
        print("\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--file-path",
        default='../data/spec_bench/model_answer/vicuna-7b-v1.5-ptp-g70+p.jsonl',
        # default='../data/spec_bench/model_answer/vicuna-7b-v1.5-ptp-float16-temp-0.7-m5-1.jsonl',
        # default='../data/spec_bench/model_answer/vicuna-7b-v1.3-sps-68m-float16-temp-0.7-m5.jsonl',
        # default='../data/spec_bench/model_answer/vicuna-7b-v1.3-medusa-float16-temp-0.7-m5.jsonl',
        # default='../data/spec_bench/model_answer/vicuna-7b-v1.3-eagle-float16-temp-0.7-m5.jsonl',
        # default='../data/spec_bench/model_answer/vicuna-7b-v1.3-eagle2-float16-temp-0.7-m5.jsonl',
        # default='../data/spec_bench/model_answer/vicuna-7b-v1.3-lookahead-float16-temp-0.7-m5.jsonl',
        # default='../data/spec_bench/model_answer/vicuna-7b-v1.3-pld-float16-m5.jsonl',
        # default='../data/spec_bench/model_answer/vicuna-7b-v1.3-hydra-float16-temp-0.7-m5.jsonl',
        # default='../data/spec_bench/model_answer/vicuna-7b-v1.3-recycling-float16-temp-0.7-m5.jsonl',
        # default='../data/spec_bench/model_answer/vicuna-7b-v1.3-samd-float16-temp-0.7-m5.jsonl',
        type=str,
        help="The file path of evaluated Speculative Decoding methods.",
    )
    parser.add_argument(
        "--base-path",
        # default='../data/spec_bench/model_answer/vicuna-7b-v1.5-vanilla-float16-temp-0.7-m5.jsonl',
        default='../data/spec_bench/model_answer/vicuna-7b-v1.3-vanilla-float16-temp-0.7-m5-1.jsonl',
        # default='../data/spec_bench/old/vicuna-7b-vanilla-float16-temp-0.7-m6.jsonl',
        type=str,
        help="The file path of evaluated baseline.",
    )
    parser.add_argument(
        "--tokenizer-path",
        default='/extra/ucibdl1/jcwill/specbench/models/vicuna-7b-v1.3',
        type=str,
        help="The file path of evaluated baseline.",
    )
    parser.add_argument(
        "--mean-report",
        action="store_true",
        default=False,
        help="report mean speedup over different runs")

    args = parser.parse_args()
    if args.mean_report:
        get_mean_speedup(jsonl_file=args.file_path, jsonl_file_base=args.base_path, tokenizer_path=args.tokenizer_path)
    else:
        get_single_speedup(jsonl_file=args.file_path, jsonl_file_base=args.base_path, tokenizer_path=args.tokenizer_path)
    # correctness(jsonl_file=args.file_path, jsonl_file_base=args.base_path, tokenizer_path=args.tokenizer_path)