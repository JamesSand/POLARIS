

# read model from huggingface
export HF_TOKEN=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

export HF_HOME="$(pwd)/hf_cache"
mkdir -p "$HF_HOME"

# ds models
models=(
    "Qwen/Qwen3-8B"
)

for model_path in "${models[@]}"; do
    model_path=$model_path num_gpus=1 gpu_memory_utilization=0.95 max_length=$((1024 * 32)) bash szz_eval_gen.sh
done



