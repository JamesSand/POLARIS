# Repository Guidelines

## 项目结构与模块组织
- `deepscaler/`: Polaris 的奖励函数与工具代码。
- `evaluation/`: 评测、打分脚本与基准数据。
- `scripts/`: 训练/评测脚本与数据工具（`scripts/train`、`scripts/eval`、`scripts/data`）。
- `parquet/`: 分阶段示例数据（stage1–3）。
- `verl/`: 训练框架源码与测试/文档。
- 入口脚本：`train_with_ray.py`、`search_optimal_temperature.py`、`drop_easy_data.py`。

## 构建、测试与开发命令
- `pip install -e ./verl` 与 `pip install -e ./` 安装可编辑依赖。
- `pip install transformers==4.51.0 vllm==0.8.4 tensordict==0.6.2` 为文档推荐运行栈。
- `python scripts/eval/eval_vllm_aime24.py --model /path/to/model ...` 使用 vLLM 快速评测（输出 JSONL）。
- `./scripts/eval/eval_model_aime24.sh --model /path/to/model ...` 使用 VeRL 评测（输出 parquet）。
- `python evaluation/grade.py --file_name evaluation/results/aime24-reproduced.parquet` 进行评分。

## 代码风格与命名规范
- 代码以 Python 为主，使用 4 空格缩进。
- 脚本默认从仓库根目录执行，避免依赖相对路径混乱。
- 文件命名偏向小写 + 下划线（如 `search_optimal_temperature.py`）。
- 训练脚本按阶段命名（`stage1.sh`/`stage2.sh`/`stage3.sh`）。
- 新数据尽量沿用 `parquet/stage*/` 与 `evaluation/benchmarks/` 结构。

## 测试指南
- 测试位于 `verl/tests/`，遵循 `pytest` 约定。
- 运行局部测试：`pytest verl/tests/...`。
- 端到端测试脚本在 `verl/tests/e2e/`，通过 `run_*.sh` 执行。

## 提交与合并请求规范
- 近期提交标题为简短祈使句，如 `Update README.md`、`fix chat template`。
- 保持一行标题，聚焦主要变更。
- PR 需包含变更摘要、已运行命令，以及新增模型/数据路径说明。

## 安全与配置提示
- Ray 调试使用 `trainer.debug=True` 并插入 `breakpoint()`，另开终端执行 `ray debug`。
- 运行 vLLM 前执行 `unset VLLM_ATTENTION_BACKEND` 以匹配推荐配置。
