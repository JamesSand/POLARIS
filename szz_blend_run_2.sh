

# read model from huggingface
export HF_TOKEN=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

export HF_HOME="$(pwd)/hf_cache"
mkdir -p "$HF_HOME"


# ds models
models=(
    # "/scratch/10922/zhsha/workspace/rotation-project/replace/2-blend/output_prorl/alpha_0.000"
    # "/scratch/10922/zhsha/workspace/rotation-project/replace/2-blend/output_prorl/alpha_0.200"
    # "/scratch/10922/zhsha/workspace/rotation-project/replace/2-blend/output_prorl/alpha_0.400"
    "/scratch/10922/zhsha/workspace/rotation-project/replace/2-blend/output_prorl/alpha_0.600"
    "/scratch/10922/zhsha/workspace/rotation-project/replace/2-blend/output_prorl/alpha_0.800"
    "/scratch/10922/zhsha/workspace/rotation-project/replace/2-blend/output_prorl/alpha_1.000"
)

for model_path in "${models[@]}"; do
    model_path=$model_path num_gpus=4 max_length=$((1024 * 16)) gpu_memory_utilization=0.95 bash szz_eval_gen.sh
done








# model_path=Qwen/Qwen3-4B-Base bash szz_eval_gen.sh




