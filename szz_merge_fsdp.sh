
set -ex

# no change here
fsdp_merged_dir="$(pwd)/fsdp_merged"
model_before_path="/scratch/10922/zhsha/workspace/rotation-project/Lucky_RL/hf_models/Qwen3-4B-Base"



# if model before path not exist, exit
if [ ! -d "$model_before_path" ]; then
  model_before_path="/fast/sliu/zhizhou/workspace/rotation-project/Lucky_RL/hf_models/Qwen3-1.7B-Base"
fi

# batch to process (edit this list)
inputs=(
# "/scratch/10922/zhsha/workspace/rotation-project/Lucky_RL/ckpts_verl/stampede3-exp/qwen3-4b-base-adam-1e-6/global_step_200/actor"
# "/scratch/10922/zhsha/workspace/rotation-project/Lucky_RL/ckpts_verl/stampede3-exp/qwen3-4b-base-svd-muon-adam-1e-6/global_step_200/actor"
"/fast/sliu/zhizhou/workspace/rotation-project/Lucky_RL/ckpts_verl/debug0110/qwen3-1.7b-base-adam-1e-6-bs128-kl0.0/global_step_200/actor"
"/fast/sliu/zhizhou/workspace/rotation-project/Lucky_RL/ckpts_verl/debug0110/qwen3-1.7b-base-adam-2e-6-bs128-kl0.0/global_step_200/actor"
"/fast/sliu/zhizhou/workspace/rotation-project/Lucky_RL/ckpts_verl/debug0110/qwen3-1.7b-base-adam-3e-6-bs128-kl0.0/global_step_200/actor"
"/fast/sliu/zhizhou/workspace/rotation-project/Lucky_RL/ckpts_verl/debug0110/qwen3-1.7b-base-svd-muon-adam-1e-6-bs128-kl0.0/global_step_200/actor"
)

# 给我写一段脚本，检测这些上边这些 folder 是否都存在
for fsdp_input_path in "${inputs[@]}"; do
  if [ ! -d "$fsdp_input_path" ]; then
    echo "[ERROR] Directory does not exist: $fsdp_input_path"
    exit 1
  fi
done

echo "All folders exist. Proceeding with merging..."

# for fsdp_input_path in "${inputs[@]}"; do
#   # derive a clean output_name from .../<exp>/global_step_xxx/actor
#   exp="$(basename "$(dirname "$(dirname "$fsdp_input_path")")")"
#   step="$(basename "$(dirname "$fsdp_input_path")")"
#   output_name="${exp}-${step}"

#   fsdp_output_path="${fsdp_merged_dir}/${output_name}"

#   python -m verl.model_merger merge \
#     --backend fsdp \
#     --local_dir "${fsdp_input_path}" \
#     --target_dir "${fsdp_output_path}"

# #   python svd_decompose.py \
#     # --model_before "${model_before_path}" \
#     # --model_after "${fsdp_output_path}" \
#     # --layer 10 \
#     # --topk 0 \
#     # --out "./figs/${output_name}_layer10_q_gate_3rows.png"
# done


# set -ex

export HF_TOKEN=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
HF_USER_OR_ORG="JameSand"   # <<< 改这里

# 1) ensure all inputs exist
for d in "${inputs[@]}"; do [[ -d "$d" ]] || { echo "[ERROR] missing: $d"; exit 1; }; done

# 2) merge + create repo + upload
for d in "${inputs[@]}"; do
  exp="$(basename "$(dirname "$(dirname "$d")")")"
  step="$(basename "$(dirname "$d")")"
  name="${exp}-${step}"
  out="${fsdp_merged_dir}/${name}"

  # python -m verl.model_merger merge --backend fsdp --local_dir "$d" --target_dir "$out"

  hf repo create --token "$HF_TOKEN" --exist-ok "${HF_USER_OR_ORG}/${name}"
  hf upload --token "$HF_TOKEN" --repo-type model "${HF_USER_OR_ORG}/${name}" "$out" . \
    --commit-message "upload ${name}"
done



# 这里给我写一个把 merge 之后的 model
#  传到 huggingface 上








