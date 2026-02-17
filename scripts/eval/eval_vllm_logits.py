#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch-inference script for AIME-24:
• N = --n stochastic roll-outs per prompt, each with a unique, random seed
• roll-outs are split across --num_gpus workers
• Each GPU loads the model once and iterates over its own seed list
• Save per-trace full-vocab logits and trace-level probability metrics
"""
import os
import json
import random
import concurrent.futures
from pathlib import Path
from datetime import datetime

import argparse
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams


# --------------------------------------------------------------------------- #
#                               Argument parser                               #
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="/path/to/model")
parser.add_argument("--t", type=float, default=1.4)
parser.add_argument("--k", type=int, default=20)
parser.add_argument("--n", type=int, default=32)
parser.add_argument("--p", type=float, default=1.0)
parser.add_argument("--max_length", type=int, default=8192)
parser.add_argument("--experiment_name", type=str, default="szz-eval")
parser.add_argument("--outpath", type=str, default="evaluation/results.jsonl")
parser.add_argument("--eval_file", type=str, default="evaluation/benchmarks/aime24.parquet")
parser.add_argument("--num_gpus", type=int, default=4)
parser.add_argument("--gpu_memory_utilization", type=float, default=0.6)
parser.add_argument("--logits_outdir", type=str, default="")

args = parser.parse_args()


# --------------------------------------------------------------------------- #
#                   Original global constants / variables                     #
# --------------------------------------------------------------------------- #
NAME = args.experiment_name
N = args.n
EVAL_FILE = args.eval_file
assert Path(EVAL_FILE).exists(), f"{EVAL_FILE} does not exist"

MODEL_PATH = args.model
MAX_TOKENS = args.max_length
TEMPERATURE = args.t
TOP_P = args.p
TOP_K = args.k
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

OUT_PATH = Path(args.outpath)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

DEFAULT_LOGITS_DIR = OUT_PATH.parent / f"{OUT_PATH.stem}_trace_logits"
LOGITS_OUTDIR = Path(args.logits_outdir) if args.logits_outdir else DEFAULT_LOGITS_DIR
LOGITS_OUTDIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
#                               Helper functions                              #
# --------------------------------------------------------------------------- #
def load_samples(filepath: str):
	"""Read parquet file and return a list of prompts."""
	df = pd.read_parquet(filepath)
	samples = [
		{
			"example_id": i,
			"prompt": df.at[i, "prompt"].tolist(),
			"answer": df.at[i, "reward_model"]["ground_truth"],
		}
		for i in range(len(df))
	]
	print(f"Total unique samples: {len(samples)}")
	return samples


def split_seeds(seeds: list[int], num_workers: int):
	chunks = [[] for _ in range(num_workers)]
	for idx, s in enumerate(seeds):
		chunks[idx % num_workers].append(s)
	return chunks


def _safe_tokenizer_max_len(tokenizer, fallback: int):
	m = getattr(tokenizer, "model_max_length", None)
	if m is None:
		return fallback
	if isinstance(m, int) and 0 < m < 1_000_000:
		return min(m, fallback)
	return fallback


def render_prompt_text(tokenizer, prompt):
	if isinstance(prompt, list):
		return tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
	return str(prompt)


def compute_trace_metrics_and_save_logits(logits: torch.Tensor, response_token_ids: torch.Tensor, logits_file: Path):
	"""
	logits: [response_len, vocab_size]
	response_token_ids: [response_len]
	"""
	logits = logits.to(torch.float16).cpu()
	torch.save(logits, logits_file)

	logits_f32 = logits.float()
	log_probs = torch.log_softmax(logits_f32, dim=-1)

	seq_len = logits.shape[0]
	if response_token_ids.numel() != seq_len:
		min_len = min(response_token_ids.numel(), seq_len)
		response_token_ids = response_token_ids[:min_len]
		log_probs = log_probs[:min_len]

	token_idx = torch.arange(response_token_ids.numel())
	trace_log_probs = log_probs[token_idx, response_token_ids]
	avg_trace_logprob = trace_log_probs.mean().item() if trace_log_probs.numel() > 0 else None

	deepconf = {}
	for k in [5, 10, 20, 40]:
		k_eff = min(k, log_probs.shape[-1])
		topk_log_probs, _ = torch.topk(log_probs, k=k_eff, dim=-1)
		token_confidence = -topk_log_probs.sum(dim=-1) / k_eff
		deepconf[f"top{k_eff}"] = token_confidence.mean().item()

	return {
		"logits_path": str(logits_file),
		"response_token_len": int(logits.shape[0]),
		"avg_trace_logprob": avg_trace_logprob,
		"deepconf_avg_confidence": deepconf,
	}


def save_full_trace_logits_and_metrics(results: list[dict], logits_dir: Path):
	print(f"Loading HF model for full logits: {MODEL_PATH}")
	tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
	model = AutoModelForCausalLM.from_pretrained(
		MODEL_PATH,
		torch_dtype=torch.bfloat16,
		device_map="auto",
		trust_remote_code=True,
	)
	model.eval()

	effective_max_len = _safe_tokenizer_max_len(tokenizer, MAX_TOKENS)
	device = next(model.parameters()).device

	for idx, item in enumerate(tqdm(results, desc="Saving full trace logits+metrics")):
		prompt_text = render_prompt_text(tokenizer, item["prompt"])
		response_text = item["response"]
		full_text = prompt_text + response_text

		prompt_ids = tokenizer(
			prompt_text,
			return_tensors="pt",
			add_special_tokens=False,
			truncation=True,
			max_length=effective_max_len,
		)["input_ids"]

		full_ids = tokenizer(
			full_text,
			return_tensors="pt",
			add_special_tokens=False,
			truncation=True,
			max_length=effective_max_len,
		)["input_ids"]

		prompt_len = prompt_ids.shape[1]
		total_len = full_ids.shape[1]
		if prompt_len >= total_len:
			item["logits_path"] = None
			item["response_token_len"] = 0
			item["avg_trace_logprob"] = None
			item["deepconf_avg_confidence"] = None
			continue

		input_ids = full_ids.to(device)
		with torch.no_grad():
			outputs = model(input_ids=input_ids)
			all_logits = outputs.logits

		response_logits = all_logits[:, prompt_len - 1 : total_len - 1, :].squeeze(0)
		response_token_ids = full_ids[0, prompt_len:total_len].cpu().to(torch.long)

		logits_file = logits_dir / f"ex{item['example_id']}_seed{item['seed']}_idx{idx}.pt"
		metrics = compute_trace_metrics_and_save_logits(response_logits, response_token_ids, logits_file)

		item.update(metrics)


# --------------------------------------------------------------------------- #
#                           Worker process (one GPU)                          #
# --------------------------------------------------------------------------- #
def worker_process(args_tuple):
	samples, seed_list, gpu_id = args_tuple
	os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
	print(f"[GPU {gpu_id}] seeds={seed_list} | loading model...", flush=True)

	llm = LLM(
		model=MODEL_PATH,
		enforce_eager=True,
		gpu_memory_utilization=args.gpu_memory_utilization,
	)

	results = []
	for seed in seed_list:
		sampling = SamplingParams(
			temperature=TEMPERATURE,
			top_p=TOP_P,
			top_k=TOP_K,
			max_tokens=MAX_TOKENS,
			seed=seed,
		)
		messages = [s["prompt"] for s in samples]
		outputs = llm.chat(messages, sampling, use_tqdm=True)

		for sample, out in zip(samples, outputs):
			results.append(
				{
					"example_id": sample["example_id"],
					"prompt": sample["prompt"],
					"answer": sample["answer"],
					"seed": seed,
					"response": out.outputs[0].text,
				}
			)
	return results


# --------------------------------------------------------------------------- #
#                                   main                                      #
# --------------------------------------------------------------------------- #
def main():
	samples = load_samples(EVAL_FILE)

	random_seeds = random.sample(range(2**31 - 1), N)
	num_workers = args.num_gpus
	seed_chunks = split_seeds(random_seeds, num_workers)

	all_results = []
	args_list = [(samples, seed_chunks[gid], gid) for gid in range(num_workers)]
	with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as ex:
		futures = [ex.submit(worker_process, tup) for tup in args_list]
		for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="GPU workers"):
			all_results.extend(fut.result())

	print(f"Total generations collected: {len(all_results)}")

	save_full_trace_logits_and_metrics(all_results, LOGITS_OUTDIR)
	print(f"Saved full trace logits to {LOGITS_OUTDIR}")

	with OUT_PATH.open("w", encoding="utf-8") as f:
		for item in all_results:
			f.write(json.dumps(item, ensure_ascii=False) + "\n")
	print(f"Saved results to {OUT_PATH}")


if __name__ == "__main__":
	main()






