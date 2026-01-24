
set -ex

# no change here
fsdp_merged_dir="$(pwd)/fsdp_merged"

# 这个地方要改成 base model 的 path，对于 ds1.5b model 来说，是再 Lucky RL 的 hf model folder 下边
model_before_path="hf_models/DeepSeek-R1-Distill-Qwen-1.5B"

# 这里要放 checkpoints 的 root dir
model_roots=(
  # "/scratch/10922/zhsha/workspace/rotation-project/Lucky_RL/ckpts_verl/stampede3-exp/qwen3-4b-base-adam-1e-6"
  # "/scratch/10922/zhsha/workspace/rotation-project/Lucky_RL/ckpts_verl/stampede3-exp/qwen3-4b-base-svd-muon-adam-1e-6"
  # "/fast/sliu/zhizhou/workspace/rotation-project/Lucky_RL/ckpts_verl/debug0110/qwen3-1.7b-base-adam-1e-6-bs128-kl0.0"
  # "/fast/sliu/zhizhou/workspace/rotation-project/Lucky_RL/ckpts_verl/debug0110/qwen3-1.7b-base-adam-2e-6-bs128-kl0.0"
  # "/fast/sliu/zhizhou/workspace/rotation-project/Lucky_RL/ckpts_verl/debug0110/qwen3-1.7b-base-adam-3e-6-bs128-kl0.0"
  # "/fast/sliu/zhizhou/workspace/rotation-project/Lucky_RL/ckpts_verl/debug0110/qwen3-1.7b-base-svd-muon-adam-1e-6-bs128-kl0.0"
  "/scratch/10922/zhsha/workspace/rotation-project/Lucky_RL/ckpts_verl/stampede3-exp/qwen3-4b-base-adam-2e-6-bs128-kl0.0"
)

# 这里放要 upload 的 checkpoints 的 steps
steps=(
  200
)

# steps=(
#   40
#   80
#   120
#   160
#   200
# )

# 构建 inputs 列表
inputs=()
for root in "${model_roots[@]}"; do
  for step in "${steps[@]}"; do
    inputs+=("${root}/global_step_${step}/actor")
  done
done

# echo "done"

# 检查这些 folder 是不是都存在
for fsdp_input_path in "${inputs[@]}"; do
  if [ ! -d "$fsdp_input_path" ]; then
    echo "[ERROR] Directory does not exist: $fsdp_input_path"
    exit 1
  fi
done

echo "All folders exist. Proceeding with merging..."

export HF_TOKEN=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
HF_USER_OR_ORG="JameSand"   # 这里是 hf username

# 1) ensure all inputs exist
for d in "${inputs[@]}"; do [[ -d "$d" ]] || { echo "[ERROR] missing: $d"; exit 1; }; done

# 2) merge + create repo + upload
for d in "${inputs[@]}"; do
  exp="$(basename "$(dirname "$(dirname "$d")")")"
  step="$(basename "$(dirname "$d")")"
  name="${exp}-${step}"
  out="${fsdp_merged_dir}/${name}"

  python -m verl.model_merger merge --backend fsdp --local_dir "$d" --target_dir "$out"

  hf repo create --token "$HF_TOKEN" --exist-ok "${HF_USER_OR_ORG}/${name}"
  hf upload --token "$HF_TOKEN" --repo-type model "${HF_USER_OR_ORG}/${name}" "$out" . \
    --commit-message "upload ${name}"
done




