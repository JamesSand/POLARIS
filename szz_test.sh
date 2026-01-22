
export PYTHONPATH="/scratch/10922/zhsha/workspace/rotation-project/POLARIS:${PYTHONPATH:-}"

out_path="/scratch/10922/zhsha/workspace/rotation-project/POLARIS/eval_outputs/qwen3-4b-base-adam-1e-6-global_step_200-t1.0-maxlen4096-p0.8-k-1-20260121_190259/minerva_processed.jsonl"

python evaluation/grade.py --file_name "$out_path"

