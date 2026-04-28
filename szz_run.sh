

# read model from huggingface
export HF_TOKEN=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

export HF_HOME="$(pwd)/hf_cache"
mkdir -p "$HF_HOME"

# ds models
models=(
    # "Qwen/Qwen3-8B"
    "Qwen/Qwen3-1.7B-Base"
)

max_length=$((1024 * 4))

for model_path in "${models[@]}"; do
    model_path=$model_path num_gpus=2 gpu_memory_utilization=0.95 max_length=$max_length bash szz_eval_gen.sh
done



