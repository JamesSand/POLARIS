#!/usr/bin/env bash
# Math eval used for the SFT-U/V sigma-interpolation experiment (Fig-3 mirror).
# Runs the 5 datasets for ONE model on ONE GPU with the exact parameters used
# in replace/RESULTS_sft_uv_sigma_interp.md, then grades each.
#
# Usage: run_fig3_math.sh <GPU_ID> <MODEL_PATH> <OUT_DIR>
#   e.g. run_fig3_math.sh 0 /path/to/models/alpha_0.4 ./eval_outputs/alpha_0.4
#
# Requires: vllm>=0.8.x env with pandas/pyarrow/datasets (see the handoff notes
# in replace/RESULTS_sft_uv_sigma_interp.md). Run from the POLARIS repo root.
set -u
GPU=${1:?gpu id}
MODEL_PATH=${2:?model path}
OUT_DIR=${3:?output dir}

PY=${PY:-python}
mkdir -p "$OUT_DIR"

export HF_HOME=${HF_HOME:-/shared/huggingface}
export CUDA_VISIBLE_DEVICES=$GPU
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export PYTHONPATH=$(pwd)

DATASETS=(
  "evaluation/benchmarks/processed/aime24_processed.parquet|32"
  "evaluation/benchmarks/processed/aime25_processed.parquet|32"
  "evaluation/benchmarks/processed/amc23_processed.parquet|8"
  "evaluation/benchmarks/processed/minerva_processed.parquet|4"
  "evaluation/benchmarks/processed/olympiad_processed.parquet|4"
)

for item in "${DATASETS[@]}"; do
  eval_file="${item%%|*}"
  n="${item##*|}"
  eval_name="$(basename "$eval_file" .parquet)"
  out_path="$OUT_DIR/${eval_name}.jsonl"

  if [[ ! -f "$out_path" ]]; then
    echo "=== $(date) | $eval_name | n=$n ==="
    $PY scripts/eval/eval_vllm.py \
        --model "$MODEL_PATH" \
        --max_length 32768 \
        --n "$n" --t 1.0 --p 0.8 --k -1 \
        --eval_file "$eval_file" \
        --num_gpus 1 \
        --gpu_memory_utilization 0.85 \
        --outpath "$out_path" || { echo "[FAIL] $eval_name"; continue; }
  fi

  if [[ -f "$out_path" ]] && ! grep -q "Grading file: ${eval_name}.jsonl" "$OUT_DIR/grade.txt" 2>/dev/null; then
    {
      echo "Grading file: ${eval_name}.jsonl"
      $PY evaluation/grade.py --file_name "$out_path"
      echo ""
    } >> "$OUT_DIR/grade.txt" 2>&1
  fi
done
echo "=== $(date) | done. Mean@n scores in $OUT_DIR/grade.txt ==="
