# AgentDSE

**基于 LLM Agent 与 ATLAS 仿真器的硬件设计空间探索框架。**

AgentDSE 使用 ATLAS 评估硬件配置的延迟与能耗，通过 LLM Agent 分析历史实验、诊断性能瓶颈并提出下一组候选配置，在有限仿真预算内探索性能、能耗与面积之间的折中。

## 工作流程

```text
定义搜索空间 → 选择种子点 → ATLAS 仿真 → 计算目标函数
                                           ↓
                   更新最优配置 ← 仿真新配置 ← Agent 诊断与提案
                         ↓
                 达到停止条件，输出结果
```

- **多维搜索**：默认联合探索 DRAM 带宽、矩阵计算单元规模和向量计算单元规模。
- **自动 tiling**：硬件配置由 AgentDSE 选择，算子 tiling 由 ATLAS 内部搜索。
- **实验持久化**：使用 SQLite 保存实验与决策记录，支持复用已有结果。
- **可配置停止策略**：支持迭代上限、边际收益早停与 Agent 主动停止。

## 项目结构

将 ATLAS 源码放在 `agent_dse/ATLAS-MICRO-2026-main/` 下：

```text
AgentDSE/
├── agent_dse/
│   ├── ATLAS-MICRO-2026-main/  # ATLAS 源码及编译产物
│   ├── full_loop_multidim.py  # 多维 DSE 入口
│   ├── agent.py              # Agent 诊断与候选配置选择
│   ├── dims.py               # 搜索空间、baseline 与配置标识
│   ├── sim.py                # 硬件配置生成与仿真调度
│   ├── sim_worker.py         # 独立进程仿真入口
│   ├── costfn.py             # 面积、功耗与目标函数
│   ├── prompts.py            # Agent 提示词
│   ├── tools.py              # 趋势分析与配置校验
│   └── store.py              # 实验数据库
├── config/
│   ├── run.json              # 工作负载与仿真参数
│   └── space.json            # 搜索维度与固定硬件参数
├── tests/                    # 功能检查脚本
├── data/                     # 实验数据库与仿真产物
├── logs/                     # 探索过程记录
└── key.json                  # 本地 LLM 接口配置
```

## 快速开始

### 1. 安装环境并编译 ATLAS

**ATLAS 包含 C/C++ 仿真器和 Python 扩展，需要先完成编译与安装，不能仅放入源码后直接运行 AgentDSE。**

将 ATLAS 源码放入上述目录后，进入其根目录：

```bash
cd agent_dse/ATLAS-MICRO-2026-main
```

按照 **[ATLAS README：Installation](agent_dse/ATLAS-MICRO-2026-main/README.md#1-installation)** 完成环境搭建、子模块初始化、依赖安装、原生仿真器编译及 Python 绑定安装。该文档也包含 HotSpot 和 TileLang 的构建说明；具体依赖与构建步骤以 ATLAS README 为准。

ATLAS 文档推荐 Python 3.10，构建工具包括 Git、Bash、Make、CMake 和 C/C++ 编译器。**编译 ATLAS 与运行 AgentDSE 应使用同一个 Python 环境**，以保证 `atlasim` 扩展兼容。

完成 ATLAS 安装后，返回 AgentDSE 项目根目录，安装本项目的直接依赖：

```bash
cd ../..
python -m pip install requests pyyaml
```

### 2. 配置 ATLAS 路径

采用上述目录布局时，将 `agent_dse/path.py` 中的 `ATLAS` 设置为：

```python
ATLAS = ROOT / "agent_dse" / "ATLAS-MICRO-2026-main"
```

将 `agent_dse/sim_worker.py` 中的 `ATLAS_PATH` 设置为：

```python
ATLAS_PATH = str(Path(__file__).resolve().parent / "ATLAS-MICRO-2026-main")
```

仿真调度器会为子进程配置 Python 模块路径和动态库路径，`atlasim` 编译产物位于 ATLAS 的 `simulator/build/lib/`。

### 3. 配置 LLM

在项目根目录创建 `key.json`，填写兼容 OpenAI Chat Completions API 的服务配置：

```json
{
  "default": {
    "model": "your-model-name",
    "key": "your-api-key",
    "url": "https://your-api-endpoint/v1",
    "temperature": 0.2,
    "timeout": 600
  }
}
```

`url` 为 API 基础地址，客户端会追加 `/chat/completions`。`key.json` 仅用于本地配置，不应将真实密钥提交到版本库。

### 4. 检查环境并运行

以下命令均在项目根目录执行：

```bash
# 检查 LLM 接口
python tests/test_01_check_llm.py

# 运行 baseline 单点仿真，不调用 LLM
python tests/test_11_check_run.py

# 启动完整多维 DSE
python -m agent_dse.full_loop_multidim
```

首次运行先评估种子点，再进入 Agent 迭代；终端会输出每轮提案、仿真指标、cost 和当前最优配置。

## 实验配置

### 工作负载与仿真参数

编辑 `config/run.json`：

```json
{
  "model_preset": "llama3_8b",
  "batch_size": 8,
  "context_length": 4096,
  "element_size": 2,
  "timeout_s": 900,
  "num_workers": 4
}
```

| 参数 | 含义 |
| --- | --- |
| `model_preset` | ATLAS 模型预设，对应其 `configs/models/` 中的模型配置 |
| `batch_size` | 推理 batch size |
| `context_length` | 每条请求的上下文长度 |
| `element_size` | 数据元素大小，单位为字节 |
| `timeout_s` | 单个候选点的仿真超时，单位为秒 |
| `num_workers` | ATLAS tiling 搜索的并行 worker 数 |

### 搜索空间与种子点

编辑 `config/space.json`。`dims` 中每个维度通过 `name` 和 `levels` 定义离散档位，所有维度的笛卡尔积构成候选空间。

默认搜索空间为：

| 维度 | 档位 |
| --- | --- |
| `bw_gbps` | 256、512、1024、2048、4096 GB/s |
| `mac_num` | 2048、4096、6144、8192 |
| `vec_num` | 64、128、256、512、768、1024 |

共 `5 × 4 × 6 = 120` 个候选点。其他配置包括：

- `seed_n`：初始种子数量，默认为 27。
- `seed_strategy`：默认使用 `farthest`，从 baseline 开始，在归一化空间内依次选取距离已选集合最远的点。
- `core_num`：芯片核心数，默认为 16。
- `fixed_hardware`：不参与搜索、但传递给每个候选点的固定硬件参数。

baseline 取每个维度的 `levels[len(levels) // 2]`，默认对应 `bw_gbps=1024`、`mac_num=6144`、`vec_num=512`。

### 优化目标

目标函数位于 `agent_dse/costfn.py`：

```text
cost = (latency / baseline_latency)
     × (energy / baseline_energy)
     × (area / baseline_area)
```

延迟和能耗来自 ATLAS，面积由 `chip_area()` 估算。baseline 的 cost 为 1，数值越低越好。可修改 `cost()` 调整各项权重或替换优化目标，并同步更新 `config/space.json` 的 `objective` 描述及 `agent_dse/prompts.py` 中的目标说明。

## 添加额外搜索维度

搜索空间、配置标识和实验记录由 `Space` 统一管理。新增维度分为**定义档位**、**接入仿真**和**更新代价模型**三个部分。

### 示例：将 SRAM 容量加入搜索

**1. 在 `config/space.json` 的 `dims` 数组中追加：**

```json
{
  "name": "buffer_size",
  "levels": [
    {"buffer_size": 1536},
    {"buffer_size": 3072},
    {"buffer_size": 6144}
  ]
}
```

同时从 `fixed_hardware` 中移除 `buffer_size`，使其由搜索档位决定。此例中的容量单位为 KB，候选空间扩展为 `120 × 3 = 360` 个点。

**2. 确认参数传递到 ATLAS。**

在 `agent_dse/sim.py` 的 `build_system_yaml()` 中，将新参数映射到芯片 YAML。`buffer_size` 已有映射，可直接复用：

```python
arch["core"]["buffer"]["buffer_size"] = int(point["buffer_size"])
```

现有映射还包括 `frequency`、`buffer_read_bw` 和 `buffer_write_bw`。对于新的硬件字段，需要在该函数中补充对应映射；DRAM 相关参数的生成逻辑位于 `agent_dse/dramcfg.py`。仅添加档位定义并不会自动改变仿真器中的硬件配置。

**3. 更新面积和功耗模型。**

如果新维度影响面积或功耗，应在 `agent_dse/costfn.py` 中实现相应模型，并在 `build_system_yaml()` 中将功耗写入配置。例如，搜索 SRAM 容量时，应将固定的 `BUFFER_AREA` 和 buffer 功耗改为随容量变化的模型。线性可加的计算单元可参考现有 `AREA_PER_UNIT`、`POWER_PER_UNIT` 与 `unit_power()` 实现，系数应来自所采用的硬件模型。

**4. 更新 Agent 说明并验证。**

在 `agent_dse/prompts.py` 中补充新维度的物理含义和权衡关系，确保提案 JSON 的 `point` 包含所有搜索维度。然后执行：

```bash
python tests/test_09_check_dims.py
python tests/test_11_check_run.py
```

检查生成的 `chip.yaml` 和 `metrics.json`，确认新维度已作用于硬件配置。`Space` 会自动将新维度加入候选枚举、baseline、配置标识和搜索空间描述，无需为其单独编写枚举或数据库主键逻辑。

## 调整迭代预算与早停条件

编辑 `agent_dse/full_loop_multidim.py` 顶部的三个常量：

```python
MAX_ITER = 20
MARGINAL_THRESHOLD_PER = 3
PATIENCE = 5
```

| 参数 | 含义 |
| --- | --- |
| `MAX_ITER` | Agent 阶段最多执行的迭代轮数，不包含初始种子仿真 |
| `MARGINAL_THRESHOLD_PER` | 最优 cost 的相对改善阈值，单位为百分比；`3` 表示 3% |
| `PATIENCE` | 允许连续低收益的轮数 |

每轮开始时，程序比较前后两次记录的最优 cost：

```text
improvement_pct = 100 × (previous_best_cost - current_best_cost)
                      / previous_best_cost
```

当改善幅度 **小于或等于** 阈值时，低收益计数加一；超过阈值则清零。连续达到 `PATIENCE` 轮时提前结束。例如，设置阈值为 `1`、耐心值为 `8`，表示允许连续 8 轮改善不超过 1%，适合更充分的探索。

若希望仅保留迭代上限而不触发这项数值早停，可将 `PATIENCE` 设置为大于 `MAX_ITER`。Agent 仍可通过返回 `stop: true` 主动结束；候选点全部评估完成时也会停止。若需调整主动停止策略，可修改 `agent_dse/prompts.py` 和 `agent_dse/agent.py` 中的提案逻辑。

## 输出与实验管理

| 路径 | 内容 |
| --- | --- |
| `data/trials.db` | SQLite 实验记录、Agent 决策与当前最优结果 |
| `data/artifacts/<配置标签>/chip.yaml` | 候选点的芯片配置 |
| `data/artifacts/<配置标签>/system.yaml` | 系统与并行策略配置 |
| `data/artifacts/<配置标签>/metrics.json` | 延迟、能耗、利用率与 stall 等仿真指标 |
| `data/artifacts/<配置标签>/atlas.log` | ATLAS 仿真日志 |
| `data/dram_cfgs/` | 自动生成的 DRAM 配置 |
| `logs/loop_trace_multidim.jsonl` | 每轮诊断、提案、仿真结果与最优配置变化 |

重复运行同一实验时，程序复用数据库中的已有记录，并重新生成本次迭代日志。切换工作负载、固定硬件参数或搜索空间时，先归档原有 `data/` 和 `logs/`，再启动独立实验，以保持 baseline、归一化指标和结果的一致性。

## 进一步阅读

- [开发手册](DEVELOPMENT_MANUAL.md)
- [多维设计空间扩展说明](MULTIDIM_DSE.md)
- [ATLAS 文档](agent_dse/ATLAS-MICRO-2026-main/README.md)
