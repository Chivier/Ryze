# CLAUDE.md — Ryze Project

## Overview

Ryze (v0.3.0) 是一个 LLM 微调流水线框架，支持集群调度。核心流程：

1. **数据处理**：PDF → OCR → Markdown → SFT 数据集生成（LLM 驱动）
2. **两阶段训练**：SFT LoRA → 合并 → GRPO LoRA → 最终合并
3. **评估**：基准测试框架，对比微调前后模型表现

## 项目结构

```
src/ryze/
├── core/           # 任务抽象、流水线编排、Runner（Local/Distributed）
│   ├── task.py     # RyzeTask ABC, TaskStatus, TaskType, TaskResult
│   ├── pipeline.py # PipelineOrchestrator（DAG 拓扑排序、fail-fast）
│   ├── runner.py   # LocalRunner / DistributedRunner
│   └── progress.py # 进度回调
├── data/           # 数据处理模块
│   ├── ocr.py      # PDFOCRProcessor（Tesseract/GLM/PaddleOCR/DocTR/EasyOCR）
│   ├── dataset.py  # SFTDatasetGenerator
│   └── processor.py# RyzeDataProcessor（组合 OCR + 数据集生成）
├── rl/             # 训练模块
│   ├── sft_lora_trainer.py  # RyzeSFTLoRATrainer
│   ├── grpo_trainer.py      # RyzeGRPOTrainer
│   ├── lora_utils.py        # LoRAManager（配置、合并）
│   └── dataset_loader.py    # 数据集加载
├── eval/           # 评估模块
│   ├── evaluator.py   # RyzeEvaluator
│   ├── metrics.py     # 评估指标
│   └── benchmark.py   # BenchmarkRunner
├── cluster/        # Ray 分布式执行
│   ├── ray_manager.py          # RayManager（GPU 资源分配、集群连接）
│   ├── ray_execution_wrapper.py# RayExecutionWrapper（任务序列化 → Ray worker 远程执行）
│   ├── ray_job_client.py       # RayJobClient（Ray Job Submission 封装）
│   └── resource.py             # ResourceTracker, GPUInfo
├── ui/             # Gradio Web UI
├── config.py       # Pydantic 嵌套配置（RyzeConfig）
└── exceptions.py   # 异常层级（RyzeError → Task/Cluster/Config/PipelineError）

configs/            # JSON 配置文件
scripts/            # 工具脚本（smoke_test_pipeline.py, example_pipeline.py）
tests/              # pytest 测试套件
├── unit/           # 单元测试（按模块组织）
└── integration/    # 集成测试（任务生命周期）
```

## 架构核心概念

### Task 抽象

所有工作单元继承 `RyzeTask`（ABC），必须实现：
- `validate_inputs() → bool`
- `resource_requirements() → ResourceRequirement`
- `execute(inputs) → TaskResult`

生命周期：`PENDING → PREPARING → RUNNING → COMPLETED | FAILED`

### Factory 模式与 `to_config()` 协议

所有处理器通过 `.as_task()` 工厂方法将业务逻辑包装为 `RyzeTask` 子类：
```python
SFTDatasetGenerator.as_task(markdown_dir, output_path)
RyzeSFTLoRATrainer.as_task()
RyzeGRPOTrainer.as_task()
RyzeEvaluator.as_task(model_path, benchmark_name)
```

GPU 任务闭包类（`SFTTrainTask`, `GRPOTrainTask`, `EvaluationTask`）实现 `to_config()` 方法，返回序列化字典供 Ray worker 远程重建 trainer：
```python
def to_config(self) -> dict:
    return {
        "trainer_class_path": "ryze.rl.sft_lora_trainer.RyzeSFTLoRATrainer",
        "trainer_config": dict(trainer.config),
    }
```

### Pipeline 编排

`PipelineOrchestrator` 管理任务 DAG：
- `add_task(task, depends_on=[])` 注册任务及依赖
- `run(runner, fail_fast)` 拓扑排序后顺序执行
- 支持 `LocalRunner`（进程内）和 `DistributedRunner`（Ray 集群，通过 `RayExecutionWrapper` 分发 GPU 任务）

### 配置系统

Pydantic v2 嵌套模型，入口为 `RyzeConfig`：
- `RyzeConfig.from_json(path)` / `from_legacy_json(path)`（向后兼容）
- 子配置：`OCRConfig`, `DatasetConfig`, `LoRAConfig`, `SFTConfig`, `GRPOConfig`, `EvaluationConfig`, `ClusterConfig`, `UIConfig`

## 开发规范

### 命名约定

- **框架类**：`Ryze` 前缀 PascalCase（`RyzeTask`, `RyzeSFTLoRATrainer`）
- **异常类**：领域后缀（`TaskError`, `ClusterError`）
- **函数/方法**：snake_case，私有方法 `_` 前缀
- **常量**：UPPER_CASE（枚举值如 `TaskStatus.PENDING`）
- **模块文件**：snake_case.py

### 代码风格

- **行宽**：100 字符（ruff）
- **Python 版本**：≥ 3.10（使用 `str | None` 联合类型语法）
- **类型标注**：全函数签名标注，循环引用用 `TYPE_CHECKING` 守卫
- **日志**：所有模块使用 `logger = logging.getLogger(__name__)`
- **Linter**：ruff（规则 E, F, I, W）
- **Formatter**：black

### 异常处理

- 使用项目自定义异常层级（`RyzeError` 基类）
- 不吞异常，错误消息包含上下文
- 在系统边界验证输入（用户输入、外部 API）

### Commit 约定

语义前缀：`feat:`, `fix:`, `test:`, `update:`, `chore:`
```
feat: replace SwarmPilot/PyLet with Ray cluster backend
test: add comprehensive test suite with 109 tests
fix: fix reading order
```

## 测试

### 运行测试

```bash
# 全部测试
pytest

# 仅单元测试
pytest tests/unit/

# 仅集成测试
pytest tests/integration/

# 带覆盖率
pytest --cov=ryze
```

### 测试标记

- `@pytest.mark.unit` — 无外部依赖
- `@pytest.mark.integration` — 完整集成测试
- `@pytest.mark.slow` — 需要 GPU/网络

### 测试模式

- 使用 `conftest.py` 中的共享 fixture（`tmp_dir`, `sample_config`, `mock_tokenizer`, `mock_ray` 等）
- 抽象类测试：创建具体子类
- 外部服务：使用 mock（Ray, RayManager）
- 验证完整生命周期：validation → execution → result

## 依赖管理

使用 `uv` 管理 Python 依赖。构建系统为 hatchling。

```bash
# 安装（含开发依赖）
uv pip install -e ".[dev]"

# 安装集群支持
uv pip install -e ".[cluster]"

# 安装量化支持
uv pip install -e ".[quantization]"
```

## 入口点

```bash
# 启动 Web UI
python launch_app.py [--config PATH] [--host HOST] [--port PORT] [--share] [--mode local|distributed]

# 烟测（完整流水线验证：Data → SFT LoRA → GRPO）
python scripts/smoke_test_pipeline.py [--model MODEL_NAME] [--mode local|ray] [--ray-address ADDRESS] [--keep] [--work-dir DIR]

# 默认配置路径
configs/default_config_v2.json
```

## 重要注意事项

- 新增处理阶段时，必须继承 `RyzeTask` 并实现三个抽象方法，同时在处理器上提供 `.as_task()` 工厂
- GPU 任务闭包类必须实现 `to_config()` 方法，返回 `{"trainer_class_path": "...", "trainer_config": {...}}` 供 Ray worker 远程重建
- 修改配置结构时，需同步更新 `from_legacy_json()` 的兼容映射
- GPU 任务（SFT, GRPO, EVAL）通过 `DistributedRunner` + `RayExecutionWrapper` 路由到 Ray worker；CPU 任务（OCR, DATASET_GEN）本地执行
- LoRA 合并策略为阶段式：SFT LoRA → merge → GRPO LoRA → final merge，避免偏离监督目标
- GRPO 阶段必须依赖 SFT 和 Data 两个上游（`depends_on=[sft_id, data_id]`），否则收不到 `grpo_data_path`
- `DatasetGenTask` 同时生成 SFT 训练数据（`train_path`）和 RL 数据（`grpo_data_path`）
- Ray 集群连接前须设置 `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0`（在 `import ray` 之前），否则 worker 因缺少 ray 模块崩溃
- `RayManager(address=None)` 启动本地 Ray runtime；`address="auto"` 连接已有集群
