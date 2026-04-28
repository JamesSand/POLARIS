#!/usr/bin/env bash
set -e

# Zhizhou: we need to add the vllm deterministic flag here

use_eager_inference=True
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0

# Zhizhou: end

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

model_path=${model_path:-"/scratch/10922/zhsha/workspace/rotation-project/Lucky_RL/hf_models/Qwen3-4B-Base"}
model_basename=${model_basename:-$(basename "$model_path")}

max_length=${max_length:-$((1024 * 4))}
t=${t:-1.0}
p=${p:-0.8}
k=${k:--1}
num_gpus=${num_gpus:-4}
gpu_memory_utilization=${gpu_memory_utilization:-0.6}
OUT_DIR="./szz_eval_outputs/${model_basename}-t${t}-maxlen${max_length}-p${p}-k${k}"
mkdir -p "$OUT_DIR"
RESULT_TXT="${OUT_DIR}/result.txt"
touch "$RESULT_TXT"

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# 每行：eval_file|n
DATASETS=(
  "evaluation/benchmarks/processed/aime24_processed.parquet|32"
  "evaluation/benchmarks/processed/aime25_processed.parquet|32"
  "evaluation/benchmarks/processed/minerva_processed.parquet|4"
  "evaluation/benchmarks/processed/olympiad_processed.parquet|4"
  "evaluation/benchmarks/processed/amc23_processed.parquet|8"
#   "evaluation/benchmarks/processed/aime24_processed.parquet|1"
)

for item in "${DATASETS[@]}"; do
    eval_file="${item%%|*}"
    n="${item##*|}"
    eval_name="$(basename "$eval_file" .parquet)"
    out_path="${OUT_DIR}/${eval_name}.jsonl"

    # 如果存在就 continue
    if [[ -f "$out_path" ]]; then
        echo "=== $(date) | ${eval_name} already exists, skip ===" >> "$RESULT_TXT"
        continue
    fi

    echo "=== $(date) | ${eval_name} | n=${n} ===" >> "$RESULT_TXT"

    python scripts/eval/eval_vllm.py \
    --model "$model_path" \
    --max_length "$max_length" \
    --n "$n" \
    --t "$t" \
    --p "$p" \
    --k "$k" \
    --eval_file "$eval_file" \
    --num_gpus "$num_gpus" \
    --gpu_memory_utilization "$gpu_memory_utilization" \
    --outpath "$out_path"

    # python evaluation/grade.py --file_name "$out_path" >> "$RESULT_TXT" 2>&1
    # echo "" >> "$RESULT_TXT"
done
