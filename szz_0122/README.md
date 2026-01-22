# 模型合并、上传和评估脚本

此目录包含两个主要功能的脚本：

## 功能 1: 合并 FSDP Checkpoints 并上传到 Hugging Face

### 文件
- `merge_and_upload.sh` - 主脚本，自动合并和上传
- `upload_to_hf.py` - Hugging Face 上传工具

### 使用方法

1. **设置环境变量**
   ```bash
   export HF_TOKEN="your_huggingface_token"
   # 获取 token: https://huggingface.co/settings/tokens
   ```

2. **运行脚本**
   ```bash
   # 处理单个实验的所有 checkpoints
   bash szz_0122/merge_and_upload.sh /path/to/verl/ckpts_verl/experiment_name
   
   # 示例
   bash szz_0122/merge_and_upload.sh /fast/sliu/zhizhou/workspace/rotation-project/shared_folder/ckpts_verl/debug0110/qwen1.7b-sgd-reset-muon-lr-1e-2-fp64
   ```

### 功能说明
- 自动查找指定文件夹下的所有 `global_step_*/actor` checkpoints
- 使用 `verl.model_merger` 合并 FSDP checkpoints
- 自动上传到 Hugging Face（仓库名格式：`JameSand/<exp>-<step>`）
- 支持批量处理多个 checkpoints

### 配置项（可在脚本中修改）
- `fsdp_merged_dir`: 合并后的模型保存目录
- `model_before_path`: 原始基础模型路径（用于 SVD 分析）
- `hf_username`: Hugging Face 用户名（默认：JameSand）

---

## 功能 2: 评估 Hugging Face 模型

### 文件
- `eval_hf_model.sh` - 模型评估脚本

### 使用方法

```bash
# 评估 Hugging Face 上的模型
bash szz_0122/eval_hf_model.sh JameSand/model-name

# 或评估本地模型
bash szz_0122/eval_hf_model.sh /path/to/local/model

# 示例
bash szz_0122/eval_hf_model.sh JameSand/qwen1.7b-sgd-reset-muon-lr-1e-2-fp64-global_step_20
```

### 评估配置
- **Max Length**: 4096 (4k)
- **Temperature**: 1.0（可通过环境变量 `t` 修改）
- **Top-p**: 0.8（可通过环境变量 `p` 修改）
- **Top-k**: -1（可通过环境变量 `k` 修改）
- **Num GPUs**: 4（可通过环境变量 `num_gpus` 修改）

### 评估数据集
- AIME 2024 (n=32)
- AIME 2025 (n=32)
- Minerva (n=4)
- Olympiad (n=4)
- AMC 2023 (n=8)

### 输出
评估结果保存在 `./eval_outputs/<model_basename>-t<t>-maxlen<max_length>-p<p>-k<k>-<timestamp>/`：
- `result.txt` - 生成日志
- `grade.txt` - 评分结果
- `*.jsonl` - 各数据集的原始输出

### 自定义配置示例
```bash
# 使用 2 个 GPU，温度 0.7
num_gpus=2 t=0.7 bash szz_0122/eval_hf_model.sh JameSand/my-model
```

---

## 完整工作流示例

```bash
# 1. 设置 HF token
export HF_TOKEN="hf_xxxxxxxxxxxxx"

# 2. 合并并上传所有 checkpoints
bash szz_0122/merge_and_upload.sh /path/to/ckpts_verl/experiment

# 3. 评估上传的模型
bash szz_0122/eval_hf_model.sh JameSand/experiment-global_step_200

# 4. 查看结果
cat eval_outputs/experiment-global_step_200-*/grade.txt
```

---

## 注意事项

1. **环境要求**
   - 需要安装 `huggingface_hub`: `pip install huggingface_hub`
   - 需要配置 `verl` 环境用于模型合并
   - 需要配置 vLLM 用于评估

2. **存储空间**
   - 合并的模型会保存在 `/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/fsdp_converted/`
   - 确保有足够的磁盘空间

3. **并行处理**
   - 评估脚本使用多 GPU 并行
   - 合并和上传脚本按顺序处理多个 checkpoints

4. **错误处理**
   - 脚本使用 `set -e`，遇到错误会立即停止
   - 如果某个 checkpoint 失败，后续的不会被处理
