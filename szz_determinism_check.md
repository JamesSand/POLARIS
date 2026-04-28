# vLLM 推理 bitwise 确定性测试

目标：在同一台机器上跑 `szz_run.sh` 两次，验证两次输出的 jsonl 是不是 **bitwise 一样**（每一道题的每一条 generation 完全相同）。

## 前置

- conda 环境：`best176`
- 脚本入口：`POLARIS/szz_run.sh`
- 内部调用：`szz_eval_gen.sh` → `python scripts/eval/eval_vllm.py`
- 数据集：默认只跑 `evaluation/benchmarks/processed/aime24_processed.parquet`，n=1
- 模型：`Qwen/Qwen3-1.7B-Base`（HF 自动下载，cache 在 `./hf_cache`）
- 输出目录：`./szz_eval_outputs/Qwen3-1.7B-Base-t1.0-maxlen4096-p0.8-k-1/aime24_processed.jsonl`

`szz_eval_gen.sh` 已经设置了 vLLM 确定性相关 env：

```bash
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
```

## 跑两次的步骤

两次串行跑，**用 GPU 1 + GPU 2**（避免 GPU 0 上的别人任务）。每次的 OUT_DIR 不一样，跑完比较 jsonl。

### 1. 改 szz_run.sh 用 num_gpus=2

`szz_run.sh` 里那行：

```bash
model_path=$model_path num_gpus=1 gpu_memory_utilization=0.95 max_length=$max_length bash szz_eval_gen.sh
```

改成 `num_gpus=2`：

```bash
model_path=$model_path num_gpus=2 gpu_memory_utilization=0.95 max_length=$max_length bash szz_eval_gen.sh
```

### 2. 第一次 run

```bash
cd /ssd2/zhizhou/workspace/rotation-project/POLARIS
source /ssd2/zhizhou/miniconda/etc/profile.d/conda.sh
conda activate best176

# 干净起点
rm -rf szz_eval_outputs/Qwen3-1.7B-Base-t1.0-maxlen4096-p0.8-k-1

CUDA_VISIBLE_DEVICES=1,2 bash szz_run.sh 2>&1 | tee /tmp/szz_run1.log

# 把 run1 输出搬走
mv szz_eval_outputs/Qwen3-1.7B-Base-t1.0-maxlen4096-p0.8-k-1 \
   szz_eval_outputs/Qwen3-1.7B-Base-t1.0-maxlen4096-p0.8-k-1__run1
```

### 3. 第二次 run

```bash
CUDA_VISIBLE_DEVICES=1,2 bash szz_run.sh 2>&1 | tee /tmp/szz_run2.log

mv szz_eval_outputs/Qwen3-1.7B-Base-t1.0-maxlen4096-p0.8-k-1 \
   szz_eval_outputs/Qwen3-1.7B-Base-t1.0-maxlen4096-p0.8-k-1__run2
```

### 4. 比较两次输出的 jsonl

两种方式，任选其一：

**(a) bitwise 比较整个文件**

```bash
F1=szz_eval_outputs/Qwen3-1.7B-Base-t1.0-maxlen4096-p0.8-k-1__run1/aime24_processed.jsonl
F2=szz_eval_outputs/Qwen3-1.7B-Base-t1.0-maxlen4096-p0.8-k-1__run2/aime24_processed.jsonl

sha256sum "$F1" "$F2"
diff "$F1" "$F2" && echo "==> BITWISE IDENTICAL" || echo "==> DIFFERENT"
```

**(b) 逐条 generation 比（更细粒度，能告诉你哪一道题不一样）**

eval_vllm.py 输出每条记录大概是 `{"example_id":..., "seed":..., "generation":..., "answer":...}`。如果整个文件 bitwise 不等但你想看是哪一条不等，可以用：

```bash
python - <<'PY'
import json
f1 = "szz_eval_outputs/Qwen3-1.7B-Base-t1.0-maxlen4096-p0.8-k-1__run1/aime24_processed.jsonl"
f2 = "szz_eval_outputs/Qwen3-1.7B-Base-t1.0-maxlen4096-p0.8-k-1__run2/aime24_processed.jsonl"

def load(p):
    with open(p) as f:
        return [json.loads(l) for l in f]

a, b = load(f1), load(f2)
print(f"run1 records: {len(a)}, run2 records: {len(b)}")

# 按 (example_id, seed) 配对
def key(r):
    return (r.get("example_id"), r.get("seed"))

ma = {key(r): r for r in a}
mb = {key(r): r for r in b}

assert ma.keys() == mb.keys(), f"key mismatch: {ma.keys() ^ mb.keys()}"

n_diff = 0
for k in ma:
    if ma[k] != mb[k]:
        n_diff += 1
        if n_diff <= 5:
            print(f"DIFF at {k}")
            for fld in ("generation", "answer"):
                if ma[k].get(fld) != mb[k].get(fld):
                    print(f"  field {fld!r}:")
                    print(f"    run1: {repr(ma[k].get(fld))[:200]}")
                    print(f"    run2: {repr(mb[k].get(fld))[:200]}")
print(f"\nTotal diffs: {n_diff} / {len(ma)}")
PY
```

## 期望

如果 vLLM 的确定性 flag 都生效，sha256 应该完全一致，diff 没有任何输出。

## 注意事项 / 已知坑

- **同一台机器**：不同机器（GPU 型号、驱动、CUDA、cuBLAS、vllm 版本任一不同）很可能不 bitwise 一致，仅同硬件 + 同环境才能比较。
- **不要并行跑**：两次必须串行（脚本顺序跑），保证 GPU 在两次之间是干净状态，避免内存碎片影响 cuBLAS 选择 kernel。
- **num_gpus 要相同**：两次 run 的 tensor parallelism 必须一致。这里都用 `num_gpus=2`。
- **model 已存在 hf_cache**：第一次 run 会下载，第二次会命中 cache，时间不同但内容应一致。
- 如果用更大的 dataset（比如解开 `szz_eval_gen.sh` 里的其他几行），`n` 也要保持一致，且 seed 是从 `random_seeds` 派生的，需保证 `PYTHONHASHSEED=0` 起作用（已在 eval_gen.sh 里设了）。

## 当前这台机器的特殊情况（仅记录，新机器忽略）

- GPU 3 的 nvml handle 取不到（`Unable to determine the device handle for GPU3 ... Unknown Error`），vLLM 在 `cuda.py:log_warnings` 模块加载时会枚举所有 4 个 device 并 crash。
- 在新机器上不需要任何 patch；如果新机器也有坏 GPU，可以在 `vllm/platforms/cuda.py` 的 `log_warnings` 外包一层 `try/except pynvml.NVMLError: pass`。
