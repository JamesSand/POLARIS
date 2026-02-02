

grading 的 code 在这里

evaluation/grade.py

和原始的 polaris 相比只修复了两个地方
1 原始的 grading 只支持 30 question 的 eval（只支持 aime 的 eval），我扩展到了支持任意数量的 question
2 原始的 grading 里边只支持一个 gt，现在改成了支持一个 list 传进来的 gt


这个 minerva 和 olympiad ground truth 是 list

这个 list 还不能 json serialization

所以现在的解法是，存在 parquet 里边的时候用 str 的格式存，

在 grade score 的时候，把这个东西转换回 list，只要 match 这个 list 里边的一个答案，就算对



0122


现在需要写两个脚本
1 给定一个 verl ckpt folder，我需要一个脚本能够把 fsdp 的 checkpoints 给我合并，并且给我上传到 hf 上去

verl 合并的脚本参考这个
```

set -ex

# no change here
fsdp_merged_dir="/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/fsdp_converted"
model_before_path="/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/hf_models/Qwen3-1.7B-Base"

# batch to process (edit this list)
inputs=(
#   "/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/ckpts_verl/debug0110/qwen1.7b-sgd-reset-muon-lr-1e-2-fp64/global_step_20/actor"
#   "/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/ckpts_verl/debug0110/qwen1.7b-sgd-reset-muon-lr-1e-2-fp64/global_step_60/actor"
  # "/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/ckpts_verl/debug0110/qwen1.7b-sgd-reset-muon-lr-1e-4-fp64/global_step_100/actor"
#   "/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/ckpts_verl/debug0110/qwen1.7b-sgd-svd-muon-lr-1e-2-fp64/global_step_20/actor"
#   "/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/ckpts_verl/debug0110/qwen1.7b-sgd-svd-muon-lr-1e-2-fp64/global_step_60/actor"
#   "/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/ckpts_verl/debug0110/qwen3-1.7b-base-sgd-lr1e-2-kl-losscoef0.001-20260111_033800-g134/global_step_50/actor"
  # "/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/ckpts_verl/debug0110/qwen3-1.7b-base-sgd-lr1e-2-kl-losscoef0.001-20260111_033800-g134/global_step_100/actor"
  # "/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/ckpts_verl/debug0110/qwen3-1.7b-base-sgd-lr1e-2-kl-losscoef0.001-20260111_033800-g134/global_step_150/actor"
  # "/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/ckpts_verl/debug0110/qwen3-1.7b-base-sgd-lr1e-2-kl-losscoef0.001-20260111_033800-g134/global_step_200/actor"
  "/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/ckpts_verl/debug0110/qwen1.7b-adam-reset-muon-lr-1e-6-fp64/global_step_200/actor"
  "/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/ckpts_verl/debug0110/qwen1.7b-adam-reset-muon-lr-1e-6-fp64/global_step_20/actor"
)

for fsdp_input_path in "${inputs[@]}"; do
  # derive a clean output_name from .../<exp>/global_step_xxx/actor
  exp="$(basename "$(dirname "$(dirname "$fsdp_input_path")")")"
  step="$(basename "$(dirname "$fsdp_input_path")")"
  output_name="${exp}-${step}"

  fsdp_output_path="${fsdp_merged_dir}/${output_name}"

  python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${fsdp_input_path}" \
    --target_dir "${fsdp_output_path}"

  python svd_decompose.py \
    --model_before "${model_before_path}" \
    --model_after "${fsdp_output_path}" \
    --layer 10 \
    --topk 0 \
    --out "./figs/${output_name}_layer10_q_gate_3rows.png"
done
```

上传 hf 的脚本参考这个
```python
import os
from huggingface_hub import HfApi

# get your token here
# https://huggingface.co/settings/tokens

local_folder = "/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/fsdp_converted/llama-muon-muonlr1e-4-spectral_norm-muonadamlr1e-6-20260110_005142-global_step_200"

basename = os.path.basename(local_folder)

api = HfApi(token=os.getenv("HF_TOKEN"))

repo_id = f"JameSand/{basename}"

api.create_repo(
    repo_id=repo_id,
    repo_type="model",
    private=False,   # 按需：True/False
    exist_ok=True   # 关键：已存在就跳过
)


api.upload_folder(
    folder_path=local_folder,
    repo_id=repo_id,
    repo_type="model",
)
```



2 给定一个 hf 上的 model，我需要一个能够直接用它在 4k len 上做 eval。然后把结果给我搞出来

参考 szz eval gen.sh 和 szz grade.sh



