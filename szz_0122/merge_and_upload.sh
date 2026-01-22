#!/usr/bin/env bash
set -euo pipefail
set -x

###############################################################################
# Hard-coded config (edit here only)
###############################################################################

# 1) Which experiment directories to process (root folders that contain global_step_*/actor)
# Put as many as you want. NO CLI args needed.
EXPERIMENT_DIRS=(
  "/scratch/10922/zhsha/workspace/rotation-project/Lucky_RL/ckpts_verl/stampede3-exp/qwen3-4b-base-adam-1e-6"
  "/scratch/10922/zhsha/workspace/rotation-project/Lucky_RL/ckpts_verl/stampede3-exp/qwen3-4b-base-svd-muon-adam-1e-6"
  # "/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/ckpts_verl/debug0110/qwen1.7b-adam-lr-2e-6-fp64"
  # "/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/ckpts_verl/another_exp/some_run"
)

# 2) Only process these global steps (numbers). Example: only 100 and 200.
# If empty -> will error (to avoid accidental uploading everything).
# STEPS_TO_UPLOAD=(100 200)
STEPS_TO_UPLOAD=(40 80 120 160)

# 3) Where to write merged FSDP outputs
FSDP_MERGED_DIR="/scratch/10922/zhsha/workspace/rotation-project/POLARIS/fsdp_merged"

# 4) HF settings
HF_USERNAME="JameSand"
HF_REPO_PREFIX="stampede3-"    # e.g. "polaris-" or "" (empty means no prefix)
HF_REPO_SUFFIX=""    # e.g. "-rl" or "" (empty means no suffix)

# 5) Upload behavior
# If true, will re-upload even if repo has same files (still commits).
# If false, we still run upload; Hugging Face will deduplicate by hash anyway.
UPLOAD_COMMIT_MSG_PREFIX="upload merged checkpoint"

# 6) Merge command (verl)
# If you need extra flags, add them here.
MERGE_BACKEND="fsdp"

###############################################################################
# Safety checks
###############################################################################

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "ERROR: huggingface-cli not found. Install: pip install -U huggingface_hub"
  exit 1
fi

# Token check: either you did `huggingface-cli login` OR export HF_TOKEN.
# We don't hard require HF_TOKEN because login stores token.
# But we can warn if neither is present.
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "NOTE: HF_TOKEN not set. Assuming you've run: huggingface-cli login"
fi

if [[ ${#EXPERIMENT_DIRS[@]} -eq 0 ]]; then
  echo "ERROR: EXPERIMENT_DIRS is empty. Edit the script and add dirs."
  exit 1
fi

if [[ ${#STEPS_TO_UPLOAD[@]} -eq 0 ]]; then
  echo "ERROR: STEPS_TO_UPLOAD is empty. Refusing to upload all steps by accident."
  echo "       Please set e.g. STEPS_TO_UPLOAD=(100 200)"
  exit 1
fi

mkdir -p "$FSDP_MERGED_DIR"

###############################################################################
# Helpers
###############################################################################

step_in_whitelist() {
  local s="$1"
  for x in "${STEPS_TO_UPLOAD[@]}"; do
    if [[ "$x" == "$s" ]]; then
      return 0
    fi
  done
  return 1
}

join_by() {
  local IFS="$1"; shift
  echo "$*"
}

###############################################################################
# Main
###############################################################################

echo "============================================================"
echo "Experiments to process:"
printf '  - %s\n' "${EXPERIMENT_DIRS[@]}"
echo "Steps to upload: $(join_by , "${STEPS_TO_UPLOAD[@]}")"
echo "Merged dir: $FSDP_MERGED_DIR"
echo "HF user: $HF_USERNAME"
echo "============================================================"

for verl_ckpt_folder in "${EXPERIMENT_DIRS[@]}"; do
  if [[ ! -d "$verl_ckpt_folder" ]]; then
    echo "WARNING: experiment dir not found, skip: $verl_ckpt_folder"
    continue
  fi

  exp_name="$(basename "$verl_ckpt_folder")"
  echo ""
  echo "##############################"
  echo "# Experiment: $exp_name"
  echo "# Root: $verl_ckpt_folder"
  echo "##############################"

  # For each requested step, check actor path existence, then merge+upload
  for step_num in "${STEPS_TO_UPLOAD[@]}"; do
    actor_path="${verl_ckpt_folder}/global_step_${step_num}/actor"
    if [[ ! -d "$actor_path" ]]; then
      echo "WARNING: missing actor path, skip: $actor_path"
      continue
    fi

    step_dir="global_step_${step_num}"
    output_name="${exp_name}-${step_dir}"
    fsdp_output_path="${FSDP_MERGED_DIR}/${output_name}"

    # HF repo id: username / (prefix + output_name + suffix)
    repo_name="${HF_REPO_PREFIX}${output_name}${HF_REPO_SUFFIX}"
    repo_id="${HF_USERNAME}/${repo_name}"

    echo "------------------------------------------"
    echo "Processing: $output_name"
    echo "  actor_path:       $actor_path"
    echo "  merged_out:       $fsdp_output_path"
    echo "  hf_repo:          $repo_id"
    echo "------------------------------------------"

    # Step 1: merge
    echo "[1/2] Merge FSDP -> HF format"
    rm -rf "$fsdp_output_path"
    mkdir -p "$fsdp_output_path"

    python -m verl.model_merger merge \
      --backend "$MERGE_BACKEND" \
      --local_dir "$actor_path" \
      --target_dir "$fsdp_output_path"

    # Step 2: create repo (exist_ok behavior via || true)
    echo "[2/2] Create repo (if not exists) and upload"
    hf repo create "$repo_id" \
      --type model 

    hf upload \
      "$repo_id" \
      "$fsdp_output_path" \
      --repo-type model \
      --commit-message "${UPLOAD_COMMIT_MSG_PREFIX}: ${output_name}"

    echo "DONE: $output_name"
    echo ""
  done
done

echo "============================================================"
echo "All done."
echo "============================================================"
