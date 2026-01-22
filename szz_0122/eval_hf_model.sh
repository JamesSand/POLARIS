#!/usr/bin/env bash
set -e

# Usage:
#   bash eval_hf_model.sh <hf_model_path_or_id>
# Example:
#   bash eval_hf_model.sh JameSand/qwen1.7b-sgd-reset-muon-lr-1e-2-fp64-global_step_20
#   bash eval_hf_model.sh /path/to/local/model

if [ $# -eq 0 ]; then
  echo "Usage: $0 <hf_model_path_or_id>"
  echo "Example: $0 JameSand/qwen1.7b-sgd-reset-muon-lr-1e-2-fp64-global_step_20"
  echo "         $0 /path/to/local/model"
  exit 1
fi

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# Model configuration
model_path="$1"
model_basename=$(basename "$model_path")

# Evaluation configuration
max_length=${max_length:-$((1024 * 4))}  # 4k length
t=${t:-1.0}
p=${p:-0.8}
k=${k:--1}
num_gpus=${num_gpus:-4}

# Output directory
timestamp="$(date +"%Y%m%d_%H%M%S")"
OUT_DIR="./eval_outputs/${model_basename}-t${t}-maxlen${max_length}-p${p}-k${k}-${timestamp}"
mkdir -p "$OUT_DIR"
RESULT_TXT="${OUT_DIR}/result.txt"
touch "$RESULT_TXT"

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "=========================================="
echo "Model Evaluation Configuration"
echo "=========================================="
echo "Model:       $model_path"
echo "Max Length:  $max_length"
echo "Temperature: $t"
echo "Top-p:       $p"
echo "Top-k:       $k"
echo "Num GPUs:    $num_gpus"
echo "Output Dir:  $OUT_DIR"
echo "=========================================="
echo ""

# Dataset configuration: eval_file|n
DATASETS=(
  "evaluation/benchmarks/processed/aime24_processed.parquet|32"
  "evaluation/benchmarks/processed/aime25_processed.parquet|32"
  "evaluation/benchmarks/processed/minerva_processed.parquet|4"
  "evaluation/benchmarks/processed/olympiad_processed.parquet|4"
  "evaluation/benchmarks/processed/amc23_processed.parquet|8"
)

# Run evaluation on each dataset
for item in "${DATASETS[@]}"; do
    eval_file="${item%%|*}"
    n="${item##*|}"
    eval_name="$(basename "$eval_file" .parquet)"
    out_path="${OUT_DIR}/${eval_name}.jsonl"

    echo "=== $(date) | ${eval_name} | n=${n} ===" | tee -a "$RESULT_TXT"

    python scripts/eval/eval_vllm.py \
      --model "$model_path" \
      --max_length "$max_length" \
      --n "$n" \
      --t "$t" \
      --p "$p" \
      --k "$k" \
      --eval_file "$eval_file" \
      --num_gpus "$num_gpus" \
      --outpath "$out_path"

    echo "Completed: $eval_name -> $out_path"
    echo ""
done

# Grade all results
echo "=========================================="
echo "Grading Results"
echo "=========================================="
GRADE_TXT="${OUT_DIR}/grade.txt"

for jsonl_file in "$OUT_DIR"/*.jsonl; do
  file_basename=$(basename "$jsonl_file")
  echo "Grading file: $file_basename" | tee -a "$GRADE_TXT"
  python evaluation/grade.py --file_name "$jsonl_file" | tee -a "$GRADE_TXT"
  echo "" | tee -a "$GRADE_TXT"
done

echo "=========================================="
echo "Evaluation completed!"
echo "Results saved to: $OUT_DIR"
echo "  - result.txt: Generation logs"
echo "  - grade.txt:  Grading scores"
echo "  - *.jsonl:    Raw evaluation outputs"
echo "=========================================="
