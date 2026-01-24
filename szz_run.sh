

# read model from huggingface
export HF_TOKEN=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

export HF_HOME="$(pwd)/hf_cache"
mkdir -p "$HF_HOME"

# qwen models
models=(
    "/scratch/10922/zhsha/workspace/rotation-project/Lucky_RL/hf_models/Qwen3-1.7B-Base"
    "/scratch/10922/zhsha/workspace/rotation-project/POLARIS/fsdp_merged/qwen3-4b-base-adam-2e-6-bs128-kl0.0-global_step_200"
)

for model_path in "${models[@]}"; do
    model_path=$model_path num_gpus=3 bash szz_eval_gen.sh
done

# ds models
models=(
    "/scratch/10922/zhsha/workspace/rotation-project/Lucky_RL/hf_models/DeepSeek-R1-Distill-Qwen-1.5B"
    "JameSand/ds-adam-1e-6-global_step_200"
    "JameSand/ds-svd-muon-adam-1e-6-global_step_200"
    "JameSand/ds-adam-2e-6-global_step_200"
    "JameSand/ds-adam-3e-6-global_step_200"
)

for model_path in "${models[@]}"; do
    model_path=$model_path num_gpus=3 max_length=$((1024 * 9)) bash szz_eval_gen.sh
done






# model_path=Qwen/Qwen3-4B-Base bash szz_eval_gen.sh




