#!/usr/bin/env bash

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# 找到 folder 下边的所有子文件夹
EVAL_OUTPUTS_DIR="/scratch/10922/zhsha/workspace/rotation-project/POLARIS/eval_outputs"
mapfile -t FOLDERS < <(find "$EVAL_OUTPUTS_DIR" -mindepth 1 -maxdepth 1 -type d | sort)


# FOLDERS=(
# #   /scratch/10922/zhsha/workspace/rotation-project/POLARIS/eval_outputs/qwen3-4b-base-adam-1e-6-global_step_200-t1.0-maxlen4096-p0.8-k-1-20260121_211349
#   /scratch/10922/zhsha/workspace/rotation-project/POLARIS/eval_outputs/qwen3-4b-base-adam-1e-6-global_step_200-t1.0-maxlen8192-p0.8-k-1-20260121_213428
#   /scratch/10922/zhsha/workspace/rotation-project/POLARIS/eval_outputs/qwen3-4b-base-svd-muon-adam-1e-6-global_step_200-t1.0-maxlen4096-p0.8-k-1-20260121_212538
# )

for d in "${FOLDERS[@]}"; do
  RESULT_TXT="$d/grade.txt"
#   如果存在就删除
  if [ -f "$RESULT_TXT" ]; then
    rm "$RESULT_TXT"
  fi

  for f in "$d"/*.jsonl; do
    file_basename=$(basename "$f")
    echo "Grading file: $file_basename" >> "$RESULT_TXT"
    python evaluation/grade.py --file_name "$f" >> "$RESULT_TXT" 2>&1
    echo "" >> "$RESULT_TXT"
  done
done


