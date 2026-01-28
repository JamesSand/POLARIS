

# read model from huggingface
export HF_TOKEN=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

export HF_HOME="$(pwd)/hf_cache"
mkdir -p "$HF_HOME"


# ds models
models=(
    # "/scratch/10922/zhsha/workspace/rotation-project/Lucky_RL/hf_models/DeepSeek-R1-Distill-Qwen-1.5B"
    # "JameSand/ds-adam-1e-6-global_step_200"
    # "JameSand/ds-svd-muon-adam-1e-6-global_step_200"
    "JameSand/ds-adam-3e-6-global_step_200"
    # "JameSand/ds-adam-1e-6-global_step_180"
    # "JameSand/ds-svd-muon-adam-1e-6-global_step_180"
    # "JameSand/ds-adam-1e-6-global_step_160"
    # "JameSand/ds-svd-muon-adam-1e-6-global_step_160"
    # "JameSand/ds-adam-1e-6-global_step_140"
    # "JameSand/ds-svd-muon-adam-1e-6-global_step_140"
    # "JameSand/ds-adam-1e-6-global_step_120"
    # "JameSand/ds-svd-muon-adam-1e-6-global_step_120"
    # "JameSand/ds-adam-1e-6-global_step_100"
    # "JameSand/ds-svd-muon-adam-1e-6-global_step_100"
    # "JameSand/ds-adam-1e-6-global_step_80"
    # "JameSand/ds-svd-muon-adam-1e-6-global_step_80"
    # "JameSand/ds-adam-1e-6-global_step_60"
    # "JameSand/ds-svd-muon-adam-1e-6-global_step_60"
    # "JameSand/ds-adam-1e-6-global_step_40"
    # "JameSand/ds-svd-muon-adam-1e-6-global_step_40"
    # "JameSand/ds-adam-1e-6-global_step_20"
    # "JameSand/ds-svd-muon-adam-1e-6-global_step_20"
)

for model_path in "${models[@]}"; do
    model_path=$model_path num_gpus=3 max_length=$((1024 * 9)) bash szz_eval_gen.sh
done

# qwen models
models=(
    # "/scratch/10922/zhsha/workspace/rotation-project/Lucky_RL/hf_models/Qwen3-1.7B-Base"
    # "/scratch/10922/zhsha/workspace/rotation-project/POLARIS/fsdp_merged/qwen3-4b-base-adam-2e-6-bs128-kl0.0-global_step_200"
    # "JameSand/qwen3-4b-base-adam-1e-6-bs128-kl0.0-global_step_200"
    # "JameSand/qwen3-1.7b-base-svd-muon-adam-2e-6-adamlr-2e-6-bs128-kl0.0-global_step_200"
    # "JameSand/qwen3-1.7b-base-svd-muon-adam-2e-6-bs128-kl0.0-global_step_200"
    # "JameSand/qwen3-1.7b-base-svd-muon-adam-3e-6-bs128-kl0.0-global_step_200"
    "JameSand/qwen3-1.7b-base-svd-muon-adam-1e-6-bs128-kl0.0-global_step_200"
    "JameSand/qwen3-1.7b-base-adam-1e-6-bs128-kl0.0-global_step_200"
    # 180
    "JameSand/qwen3-1.7b-base-svd-muon-adam-1e-6-bs128-kl0.0-global_step_180"
    "JameSand/qwen3-1.7b-base-adam-1e-6-bs128-kl0.0-global_step_180"
    # 160
    "JameSand/qwen3-1.7b-base-svd-muon-adam-1e-6-bs128-kl0.0-global_step_160"
    "JameSand/qwen3-1.7b-base-adam-1e-6-bs128-kl0.0-global_step_160"
    # 140
    "JameSand/qwen3-1.7b-base-svd-muon-adam-1e-6-bs128-kl0.0-global_step_140"
    "JameSand/qwen3-1.7b-base-adam-1e-6-bs128-kl0.0-global_step_140"
    # 120
    "JameSand/qwen3-1.7b-base-svd-muon-adam-1e-6-bs128-kl0.0-global_step_120"
    "JameSand/qwen3-1.7b-base-adam-1e-6-bs128-kl0.0-global_step_120"
    # 100
    "JameSand/qwen3-1.7b-base-svd-muon-adam-1e-6-bs128-kl0.0-global_step_100"
    "JameSand/qwen3-1.7b-base-adam-1e-6-bs128-kl0.0-global_step_100"
    # 80
    "JameSand/qwen3-1.7b-base-svd-muon-adam-1e-6-bs128-kl0.0-global_step_80"
    "JameSand/qwen3-1.7b-base-adam-1e-6-bs128-kl0.0-global_step_80"
    # 60
    "JameSand/qwen3-1.7b-base-svd-muon-adam-1e-6-bs128-kl0.0-global_step_60"
    "JameSand/qwen3-1.7b-base-adam-1e-6-bs128-kl0.0-global_step_60"
    # 40
    "JameSand/qwen3-1.7b-base-svd-muon-adam-1e-6-bs128-kl0.0-global_step_40"
    "JameSand/qwen3-1.7b-base-adam-1e-6-bs128-kl0.0-global_step_40"
    # 20
    "JameSand/qwen3-1.7b-base-svd-muon-adam-1e-6-bs128-kl0.0-global_step_20"
    "JameSand/qwen3-1.7b-base-adam-1e-6-bs128-kl0.0-global_step_20"
)

for model_path in "${models[@]}"; do
    model_path=$model_path num_gpus=3 bash szz_eval_gen.sh
done







# model_path=Qwen/Qwen3-4B-Base bash szz_eval_gen.sh




