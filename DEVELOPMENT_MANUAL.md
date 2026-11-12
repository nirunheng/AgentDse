# AgentDSE 开发手册

> 照着这份文档从上往下做，每一步都有：**创建什么文件 / 写什么代码 / 怎么验证这一步真的成了**。
> 不要跳步。Step 2 是前置，不做过不了 Step 3。

---

## 0. 已实测的环境事实（不用再猜）

这些数字和结论是我在这台机器上真跑出来的，直接采信：

| 项目 | 实测值 |
|---|---|
| 机型 | 64 核，Python 3.12.4，`/opt/anaconda3/bin/python3` |
| ATLAS 库 | `ATLAS-MICRO-2026-main/simulator/build/lib/atlasim.cpython-312-x86_64-linux-gnu.so` |
| 单次 cloud inference 耗时 | **31 ~ 60 s**（`num_workers=4` 实测 31.5 s；`=1` 时 48~60 s；ch=2 最慢 102 s） |
| 每个点必须独立进程 | 同进程连跑两点会差 **0.4%**（`9642` vs `9683` cycles） |
| 独立进程的确定性 | **逐位相同**：`9642 / 9642 / 9642` cycles |
| 一次仿真的输出 | `latency=3.34144e-4 s`、`energy=4.017e-2 J`、`matrix_util=0.1991`、`dram_util=0.6882`、`dram_stall=4178 cycles` |
| 参考配置 | `core_num=16`、`llama3_8b`、`batch=8`、`context=4096`、DRAM=`cloud_1TBps` |

**结论：4 个种子 + 2 个 Agent 决策 ≈ 6 × 40 s ≈ 4 分钟；就算 7 点全跑也不到 10 分钟。**

三个必须知道的坑（Step 2 会讲前两个）：

1. `multiprocessing.cpu_count` 必须限制，否则 64 核 → 默认 51 个 spawn 进程 → **挂死**
2. `str(SimulatorOperatorType.GEMM)` 返回的是 `"SimulatorOperatorType.GEMM"` 而不是 `"gemm"` → C++ 端直接 `exit(-1)` → **挂死**
3. 每个候选点**必须用独立子进程**跑。booksim 的 `rand()` 只在 `noc_main()` 里 `srand(1234)`，走库调用路径不会重新播种，同进程连跑第二点就消耗了共享的 RNG 状态 → 0.4% 假波动。这会让你的“边际收益 < 阈值”误判。

第 1、2 条是之前 `runs/atlas_dse_test/iter_000/atlas_exec.log` 停在 `Processing operator qkv_proj` 的原因。**不是死锁，是 C++ 收到非法算子类型后退出，父进程在等 Pool，于是看着像挂死。**

---

## 0.5 三个实测结论（会决定你的实验是否成立，务必先读）

我用完整 e2e 仿真（`llama3_8b`, batch=8, ctx=4096, core_num=16）把两条可调轴都扫了一遍。结论很硬，直接影响代价函数怎么定义。

### 结论 A：`channel` 必须是 2 的幂，否则点会重复

地址映射器 `OneLevelInterleave` 用 `calc_log2(channel)` 算通道位宽（`linear_mappers.h` 里 `m_addr_bits[0] = calc_log2(count[0])`）。所以：

- `channel=5` 的仿真结果与 `channel=4` **逐位相同**
- `channel=10` 等同 `channel=8`
- `channel=20` / `24` 等同 `channel=16`

**所以旧版手册那张 20 点表里有 11 个点是假点。** 真实的独立带宽档位只有 `channel ∈ {2,4,8,16,32,64,128}`，对应 `128 / 256 / 512 / 1024 / 2048 / 4096 / 8192 GB/s`。

### 结论 B：`rate` 轴调的是延迟，不是带宽

`channel=16` 固定、只把 `rate` 从 100 提到 4000 Mbps：结果延迟从 1.17 ms 一路降到 0.20 ms，看起来像“带宽增加”。但如果同时按比例放大 timing 周期数（`mode B`，模拟真实速率分档：频率翻倍则所有周期约束也翻倍），结果**几乎完全不变**：

| rate | 保持周期数不变 (mode A) | 按比例放大周期数 (mode B，物理正确) |
|---|---|---|
| 1000 | latency `2.40e-4` | latency `3.34e-4` |
| 2000 | latency `2.06e-4` | latency `3.34e-4` |
| 4000 | latency `2.00e-4` | — |

**mode B 下 1000/2000 与 500 Mbps 的延迟一模一样。** 也就是说 `rate` 单独调只改变了“一次 DRAM 事务的绝对时间”，并没有改变稳态带宽。**不要用 `rate` 当单参数轴**，否则你测的是 DRAM 延迟而不是带宽。

### 结论 C：真正的带宽轴下，代价函数必须带面积

`channel` 扫（500 Mbps），实测：

| channel | 带宽 GB/s | latency s | energy J | `dram_util` | 功率 W |
|---|---|---|---|---|---|
| 2 | 128 | 1.4613e-3 | 0.175438 | 0.941 | 120.05 |
| 4 | 256 | 8.2554e-4 | 0.099141 | 0.889 | 120.09 |
| 8 | 512 | 4.9238e-4 | 0.059162 | 0.777 | 120.16 |
| 16 | 1024 | 3.3414e-4 | 0.040174 | 0.688 | 120.23 |
| 32 | 2048 | 2.6051e-4 | 0.031338 | 0.567 | 120.29 |
| 64 | 4096 | 2.2950e-4 | 0.027617 | 0.511 | 120.33 |
| 128 | 8192 | 2.1926e-4 | 0.026388 | 0.488 | 120.35 |

关键点：**功率全程恒定 ≈120.2 W**（能量 = 120 × 延迟，几乎正比）。所以

```
cost = (L/L0) × (E/E0) ≈ (L/L0)^2
```

是一条**单调递减、没有内部极小值**的曲线。用这个代价函数，答案永远是“通道越多越好”，Agent 没有任何决策空间，早停永远不会触发。

**必须加面积项**，否则这个实验不成立：

```
cost = (L/L0) × (E/E0) × (A/A0)     其中 A/A0 = channel/8（baseline 取 ch=8）
```

加面积后立刻出现内部最优点（实测，以 ch=8 为 1.0）：

| channel | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|---|---|
| cost | 2.2002 | 1.4048 | **1.0000** | **0.9216** | 1.1210 | 1.7406 | 3.1779 |

**这才是单峰曲线，Agent 才有东西可学。** 本手册后续全部按这个代价函数来写。

> 面积项的物理依据：DRAM 面积本就随通道数增加（`chip.yaml` 里 `dram.area` 是常量，但真实芯片不是）。如果你想更严格，可以把 `dram.area` 也改成随 channel 线性增长——那样 ATLAS 内部的功耗/面积统计会一致，但这属于改模型，**本手册不改，只在代价函数里体现**。

---

## 1. DRAM 带宽的可调机制（这决定了你能做多少个点）

读 `ATLAS-MICRO-2026-main/simulator/src/dram/ramulator2/src/dram/impl/HB-DRAM.cpp` 得到：

```
带宽 = channel 数 × 1024 pin × 500 Mbps
     = channel 数 × 512 Gbit/s
     = channel 数 × 64 GB/s
     = channel 数 × 0.064 TB/s
```

验证：`cloud_1TBps.yaml` 的 `channel: 16` → `16 × 64 = 1024 GB/s = 1.024 TB/s` ✅
（用 `Chip.get_max_dram_bw_per_core()` 实测确认，7 档全部精确等于 `channel × 64`）

**容量 = channel 数 × 每通道密度（preset 里的 Gb）**

5 个现成配置全都是 **64 Gb = 8 GB**：

| 文件 | channel | preset 密度 | 带宽 | 容量 |
|---|---|---|---|---|
| `cloud_0.25TBps.yaml` | 4 | 16 Gb | 256 GB/s | 64 Gb |
| `cloud_0.5TBps.yaml` | 8 | 8 Gb | 512 GB/s | 64 Gb |
| `cloud_1TBps.yaml` | 16 | 4 Gb | 1024 GB/s | 64 Gb |
| `cloud_2TBps.yaml` | 32 | 2 Gb | 2048 GB/s | 64 Gb |
| `cloud_4TBps.yaml` | 64 | 1 Gb | 4096 GB/s | 64 Gb |

**这是好消息：这 5 个点是纯带宽扫描，容量恒定。** 之前的担心（带宽涨容量跌）不成立。

### 真实可用候选点（只有 7 个，不是 20 个）

带宽只能靠 `channel` 拉开，而 `channel` 必须是 2 的幂（见 §0.5 结论 A）。所以：

| # | channel | 密度(Gb) | 容量(Gb) | 带宽(GB/s) |
|---|---|---|---|---|
| 0 | 2 | 16 | 32 | 128 |
| 1 | 4 | 16 | 64 | 256 |
| 2 | 8 | 8 | 64 | 512 |
| 3 | 16 | 4 | 64 | 1024 |
| 4 | 32 | 2 | 64 | 2048 |
| 5 | 64 | 1 | 64 | 4096 |
| 6 | 128 | 1 | 128 | 8192 |

> 前 7 档实测全部能跑通（`ch=2` 容量 32 Gb 也就够——实测峰值地址只有 0.16 GiB）。
> `channel_lowest_bit` 全部保持 `5`，不要动。
> `HB-DRAM.cpp` 里 `_1024pin_512col` 只有 1/2/4/8/16 Gb 五种密度，没有 32 Gb。所以 `ch=2` 最多只能给 32 Gb 容量，够用。

### 如果 7 个点不够，唯一合法的扩展方式

不要把 channel 改成非 2 的幂（那是假点）。正确做法是**在同一 channel 下换 timing preset**，但注意 §0.5 结论 B：换速率本身不改变稳态带宽，只改变延迟。所以这只能算**第二个参数**，不是“单参数”了。

两条路可选：

- **A（推荐）**：就用这 7 个点，`initial_offsets` 先跑 4 个，Agent 决策 2~3 次。7 个点对“单参数单峰曲线”完全够，且结论干净。
- **B**：把 `channel` 和 `dram_util` 类型等做成 2D，那是另一份实验，不要混进来。

> **本手册按 A 写。** 如果你坚持要 20 个点，唯一诚实的说法是“2D（channel × rate）探索”，不是单参数。

---

## 2. 目录结构总览

```
Beacon/
├── key.json                          # 已有，API 配置
└── agentdse/                         # 全部新建
    ├── DEVELOPMENT_MANUAL.md         # 本文件
    ├── config/
    │   ├── space.json                # 7 个真实候选点（channel 为 2 的幂）
    │   └── run.json                  # 运行参数（模型、batch、ctx、worker 数）
    ├── agentdse/
    │   ├── __init__.py
    │   ├── paths.py                  # 所有路径常量，集中一处
    │   ├── llm.py                    # OpenAI-compatible 客户端（用 requests）
    │   ├── store.py                  # sqlite：trials / best
    │   ├── dramcfg.py                # 生成 dram yaml
    │   ├── sim_worker.py             # 子进程入口，真正调 ATLAS
    │   ├── sim.py                    # 调度子进程、解析结果
    │   ├── tools.py                  # 4 个工具
    │   ├── prompts.py                # 2 个 prompt
    │   └── agent.py                  # 两段式决策循环
    ├── scripts/
    │   ├── 01_check_llm.py
    │   ├── 02_run_one.py
    │   ├── 03_make_seed.py
    │   ├── 04_ask_agent.py
    │   ├── 05_full_loop.py
    │   └── 06_plot.py
    ├── data/
    │   ├── trials.db                 # sqlite（自动建）
    │   ├── seed/                     # 03_make_seed.py 写出的 7 份真值
    │   ├── dram_cfgs/                # 生成的 dram yaml（每点一份）
    │   └── artifacts/                # 每次仿真的中间产物
    └── logs/                         # loop_trace.jsonl / report.png / report_table.txt
```

**需要改的 C++ / ATLAS 文件只有 3 个 Python 文件，不需要改 C++、不需要重新编译。** 见 Step 2。

---

## Step 1：API 连通性

### 1.1 确认 `key.json`

已存在。**本机的实际 profile 名是 `mimo_code_plan`，不是 `default`**：

```json
{
  "mimo_code_plan": {
    "model": "deepseek-v4-flash",
    "key": "sk-...",
    "url": "https://api.pezayo.com/v1",
    "temperature": 0.2,
    "timeout": 600
  }
}
```

`url` 写到 `/v1` 为止，不带 `/chat/completions`。

> 所以 `LLM.from_file()` 不能硬编码 `profile="default"`。下面 1.3 的代码已做兜底：
> 有 `default` 就用 `default`，否则在只有一个 profile 时自动选它。若你的 `key.json`
> 有多个 profile 且没有 `default`，显式传 `profile="mimo_code_plan"`。

### 1.2 创建 `agentdse/agentdse/paths.py`

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]      # Beacon/agentdse
BEACON = ROOT.parent                             # Beacon
ATLAS = BEACON / "ATLAS-MICRO-2026-main"
ATLAS_LIB = ATLAS / "simulator" / "build" / "lib"

CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
DRAM_DIR = DATA_DIR / "dram_cfgs"
ARTIFACT_DIR = DATA_DIR / "artifacts"
DB_PATH = DATA_DIR / "trials.db"
LOG_DIR = ROOT / "logs"

KEY_JSON = BEACON / "key.json"
```

### 1.3 创建 `agentdse/agentdse/llm.py`

```python
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import requests


class LLM:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.model = cfg["model"]
        self.key = cfg["key"]
        self.url = cfg["url"].rstrip("/")
        self.temperature = float(cfg.get("temperature", 0.2))
        self.timeout = float(cfg.get("timeout", 600))

    @classmethod
    def from_file(cls, path, profile: Optional[str] = None) -> "LLM":
        payload = json.loads(path.read_text())
        if profile is None:
            # 本机的 key.json 里 profile 名是 mimo_code_plan，不是 default。
            # 优先用 "default"；没有就退回到唯一那一个 profile。
            if "default" in payload:
                profile = "default"
            elif len(payload) == 1:
                profile = next(iter(payload))
            else:
                raise KeyError(
                    f"key.json has no 'default' profile and {len(payload)} "
                    f"candidates {list(payload)}; pass profile= explicitly"
                )
        return cls(payload[profile])

    def chat(self, system: str, user: str) -> str:
        resp = requests.post(
            f"{self.url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "temperature": self.temperature,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def chat_json(self, system: str, user: str) -> Dict[str, Any]:
        text = self.chat(system, user)
        return _extract_json(text)


def _extract_json(text: str) -> Dict[str, Any]:
    """Tolerate ```json fences and leading prose."""
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        s = s.rsplit("```", 1)[0]
    start, end = s.find("{"), s.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"no JSON object in model output: {text[:400]}")
    return json.loads(s[start : end + 1])
```

### 1.4 创建 `agentdse/scripts/01_check_llm.py`

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentdse.llm import LLM
from agentdse.paths import KEY_JSON


def main() -> int:
    print(f"key.json = {KEY_JSON}")
    llm = LLM.from_file(KEY_JSON)
    print(f"model    = {llm.model}")
    print(f"url      = {llm.url}")

    answer = llm.chat("You are a terse assistant.", "Reply with exactly the word: PONG")
    print(f"raw reply = {answer!r}")
    assert "PONG" in answer.upper(), f"unexpected reply: {answer!r}"

    payload = llm.chat_json(
        "You output only JSON.",
        'Return exactly {"ok": true, "n": 7}',
    )
    print(f"json reply = {payload}")
    assert payload.get("ok") is True and payload.get("n") == 7

    print("STEP1_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### 1.5 验证

```bash
cd /data2/nirunheng/Project/Beacon
python3 agentdse/scripts/01_check_llm.py
```

必须打印 `STEP1_OK`。看到 `401` 改 `key.json` 的 `key`；看到 `404` 检查 `url` 是否少了/多了 `/v1`；看到 `ConnectionError` 检查网络或代理。

---

## Step 2：打两个 ATLAS 补丁（前置，必须做）

### 2.1 补丁 A：算子类型的 `str()`

**症状**：`worker_output.log` 里 `Find unsupported operator type: SimulatorOperatorType.GEMM`。

**根因**：C++ 端 `simulator/src/common/task.h:41` 用
```cpp
std::string type = config["type"].as<std::string>();
if (is_computation(type)) ...
std::cerr << "Find unsupported operator type: " << type << std::endl;
exit(-1);
```
它只认 `"gemm"` / `"decode-attention"` / `"communication"` / `"spmd-general"` / `"mpmd-general"`。

而 Python 端 `tiling_explorer_cloud.py:142` 写的是
```python
"type": str(SimulatorOperatorType.GEMM),
```
本机 pybind11 2.13.6 下 `str(...)` 得到 `"SimulatorOperatorType.GEMM"`，不是 `"gemm"`。
（对比：仓库自带的 `kick_the_tires/.../comp_description.yaml` 里是 `type: gemm`，说明原作者环境里 `str()` 行为不同。）

**修法**（3 个文件，11 处，不用重新编译）：

在 `ATLAS-MICRO-2026-main/frontend/auto/` 下，把
```python
str(SimulatorOperatorType.XXX)
```
全部替换成
```python
SimulatorOperatorType.XXX.to_string()
```

一条命令搞定：

```bash
cd /data2/nirunheng/Project/Beacon/ATLAS-MICRO-2026-main
sed -i 's/str(SimulatorOperatorType\.\([A-Za-z]*\))/SimulatorOperatorType.\1.to_string()/g' \
    frontend/auto/tiling_explorer_cloud.py \
    frontend/auto/tiling_explorer_edge.py \
    frontend/auto/system.py
```

验证：

```bash
grep -c "to_string()" frontend/auto/tiling_explorer_cloud.py   # 应为 8
grep -c "to_string()" frontend/auto/tiling_explorer_edge.py    # 应为 1
grep -c "to_string()" frontend/auto/system.py                  # 应为 2
grep -c "str(SimulatorOperatorType" frontend/auto/*.py         # 应为 0
```

> 更彻底的修法是改 `simulator/src/python_bindings.cpp`，给 enum 加 `__str__`，但那要重新编译。周五前不要动。

### 2.2 补丁 B：限制 tiling worker 数

**症状**：64 核机器上默认 `int(0.8 * 64) = 51` 个 `spawn` 进程，慢到像挂死。

**根因**：`tiling_explorer_cloud.py:2267`
```python
num_workers: int = int(0.8 * multiprocessing.cpu_count()),
```
这是**默认参数**，在**模块导入时求值一次**。而 `CloudSystem.inference`（`system.py:150`）**没有** `num_workers` 参数，所以你没法从外部传。

**修法**：在**导入 `frontend.auto.*` 之前**改掉 `cpu_count`。

```python
import multiprocessing
multiprocessing.cpu_count = lambda: 6      # → 默认 worker 数 = 4
```

必须严格在 `import frontend.auto.system` 之前。放在 `sim_worker.py` 的最顶部（Step 4 会写）。

**不要在生产里改 ATLAS 源码**，用运行时覆盖，可逆且不用重新编译。

---

## Step 3：跑一次仿真，输出一个 JSON

### 3.1 创建 `agentdse/config/run.json`

```json
{
  "model_preset": "llama3_8b",
  "batch_size": 8,
  "context_length": 4096,
  "element_size": 2,
  "num_workers": 4,
  "timeout_s": 900,
  "core_num": 16,
  "fixed_hardware": {
    "mac_num": 8192,
    "vec_num": 512,
    "buffer_size": 3072,
    "buffer_read_bw": 8192,
    "buffer_write_bw": 8192,
    "frequency": 1000
  }
}
```

`core_num=16` 是**硬约束**：`tiling_explorer_cloud.py:2287` 要求 `int(sqrt(core_num))**2 == core_num` 且网格边长为偶数。16 = 4×4 ✅。**别改成 8 或 32**（8=2.83²、32=5.66²，都会 assert 失败）。

### 3.2 创建 `agentdse/agentdse/dramcfg.py`

从模板生成任意带宽的 dram yaml。

```python
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import yaml

from .paths import DRAM_DIR

TEMPLATE = {
    "Frontend": {"impl": "HBFrontend", "clock_ratio": 1},
    "MemorySystem": {
        "impl": "GenericDRAM",
        "clock_ratio": 1,
        "DRAM": {
            "impl": "HBDRAM",
            "org": {
                "preset": "HBDRAM_4Gb_1024pin_512col",
                "channel": 16,
            },
            "timing": {"preset": "HBDRAM_500Mbps"},
        },
        "Controller": {
            "impl": "Generic",
            "Scheduler": {"impl": "FRFCFS"},
            "RefreshManager": {"impl": "AllBank"},
            "RowPolicy": {"impl": "OpenRowPolicy"},
            "plugins": None,
        },
        "AddrMapper": {
            "impl": "OneLevelInterleave",
            "channel_lowest_bit": 5,
        },
    },
}

# 只使用 HB-DRAM.cpp 里真实存在的 _1024pin_512col 密度
# 注意：没有 32 Gb 的 _1024pin_512col 变体，别写 32
LEGAL_DENSITY_GB = (1, 2, 4, 8, 16)


def bw_gbps(channel: int) -> int:
    """1024 pin x 500 Mbps -> 64 GB/s per channel."""
    return channel * 64


def make_cfg(channel: int, density_gb: int) -> Dict:
    if density_gb not in LEGAL_DENSITY_GB:
        raise ValueError(f"density must be one of {LEGAL_DENSITY_GB}, got {density_gb}")
    # 关键：channel 必须是 2 的幂，否则地址映射器 calc_log2() 会静默截断，
    # 产生与更低档位逐位相同的假点（见手册 §0.5 结论 A）。
    if channel < 1 or (channel & (channel - 1)) != 0:
        raise ValueError(f"channel must be a power of 2, got {channel}")
    cfg = yaml.safe_load(yaml.safe_dump(TEMPLATE))
    cfg["MemorySystem"]["DRAM"]["org"]["preset"] = f"HBDRAM_{density_gb}Gb_1024pin_512col"
    cfg["MemorySystem"]["DRAM"]["org"]["channel"] = int(channel)
    return cfg


def write_cfg(channel: int, density_gb: int) -> Path:
    DRAM_DIR.mkdir(parents=True, exist_ok=True)
    path = DRAM_DIR / f"bw{bw_gbps(channel)}GBps.yaml"
    with path.open("w") as f:
        yaml.safe_dump(make_cfg(channel, density_gb), f, sort_keys=False)
    return path


def write_all(points: List[Dict]) -> Dict[int, str]:
    """points: [{"channel": 4, "density_gb": 16}, ...] -> {bw_gbps: path_str}"""
    out: Dict[int, str] = {}
    for p in points:
        out[bw_gbps(p["channel"])] = str(write_cfg(p["channel"], p["density_gb"]))
    return out
```

### 3.3 创建 `agentdse/agentdse/sim_worker.py`

**子进程入口。** 所有 `os.environ` / `sys.path` / `multiprocessing` 的修补都在这里，保证不污染主进程。

```python
#!/usr/bin/env python3
"""Run ONE ATLAS cloud inference. Called as a subprocess.

argv: <system_json> <artifacts_dir> <out_json>
"""
from __future__ import annotations

import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

# ---- 必须在 import frontend.* 之前 -------------------------------------
# num_workers defaults are evaluated once, at module import time.
_WORKERS = int(os.environ.get("AGENTDSE_TILING_WORKERS", "4"))
multiprocessing.cpu_count = lambda: max(2, int(_WORKERS / 0.8))
# ----------------------------------------------------------------------

ATLAS = "/data2/nirunheng/Project/Beacon/ATLAS-MICRO-2026-main"

sys.path.insert(0, ATLAS + "/simulator/build/lib")
sys.path.insert(0, ATLAS)
os.chdir(ATLAS)


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: sim_worker.py <system_json> <artifacts_dir> <out_json>", file=sys.stderr)
        return 2

    system_json, artifacts_dir, out_json = sys.argv[1:4]

    import yaml
    from frontend.hardware_parser import CloudSystemConfig
    from frontend.model_parser import get_model_config_from_hf
    from frontend.util import generate_context_slot_mapping
    from frontend.auto.system import CloudSystem

    spec = json.loads(Path(system_json).read_text())

    with open(spec["system_yaml"]) as f:
        cc = CloudSystemConfig.from_yaml(yaml.safe_load(f))

    model_config = get_model_config_from_hf(
        spec["model_preset"],
        os.path.join(ATLAS, f"configs/models/{spec['model_preset']}.json"),
    )

    ctx = [int(spec["context_length"])] * int(spec["batch_size"])
    system = CloudSystem(
        system_config=cc,
        model_config=model_config,
        element_size=int(spec["element_size"]),
    )
    slot_map = generate_context_slot_mapping(cc, ctx)

    t0 = time.time()
    result = system.inference(
        context_length_list=ctx,
        context_slot_mapping=slot_map,
        intermediate_result_dir=artifacts_dir,
    )
    elapsed = time.time() - t0

    stats = result["intra_chip_computation_performance"].e2e_stats
    payload = {
        "latency_s": float(result["latency"]),
        "energy_j": float(result["energy"]),
        "matrix_util": float(stats.matrix_util()),
        "vector_util": float(stats.vector_util()),
        "dram_util": float(stats.dram_util()),
        "buffer_util": float(stats.buffer_util()),
        "noc_util": float(stats.noc_util()),
        "dram_stall_cycles": int(stats.matrix_bubble_dram_cycles),
        "onchip_stall_cycles": int(stats.matrix_bubble_on_chip_cycles),
        "dram_cycles": int(stats.dram_cycles),
        "matrix_cycles": int(stats.matrix_cycles),
        "dram_energy_j": float(stats.dram_energy),
        "matrix_energy_j": float(stats.matrix_energy),
        "vector_energy_j": float(stats.vector_energy),
        "buffer_energy_j": float(stats.buffer_energy),
        "noc_energy_j": float(stats.noc_energy),
        "memory_access_bytes": float(stats.memory_access_bytes),
        "flop_count": float(stats.flop_count),
        "arithmetic_intensity": float(stats.arithmetic_intensity()),
        "elapsed_s": round(elapsed, 2),
    }

    Path(out_json).write_text(json.dumps(payload, indent=2))
    print(json.dumps({"ok": True, "elapsed_s": payload["elapsed_s"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### 3.4 创建 `agentdse/agentdse/sim.py`

主进程侧：生成 `system.yaml`、拉子进程、收结果、算 cost。

```python
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from . import dramcfg
from .paths import ARTIFACT_DIR, ATLAS, ATLAS_LIB, ROOT

WORKER = Path(__file__).resolve().parent / "sim_worker.py"

_CHIP_TEMPLATE = {
    "architecture": {
        "frequency": 1000,
        "core_num": 16,
        "core": {
            "controller": {"power": 1.0, "area": 1.0},
            "matrix": {"mac_num": 8192, "power": 1.1, "area": 1.1},
            "vector": {"vec_num": 512, "power": 1.2, "area": 1.2},
            "buffer": {
                "buffer_size": 3072,
                "read_bw": 8192.0,
                "write_bw": 8192.0,
                "power": 1.3,
                "area": 1.3,
            },
        },
        "dram": {"config_path": "", "power": 1.4, "area": 1.4},
        "noc": {
            "topology": "mesh",
            "config_path": "configs/architecture/noc/4x4_mesh",
            "flit_size": 64,
            "power": 1.5,
            "area": 1.5,
        },
    }
}

_SYSTEM_TEMPLATE = {
    "system": {
        "type": "cloud",
        "chip": {"config_path": ""},
        "parallelism": {
            "tp_size": 8,
            "ep_size": 8,
            "tp_scope": "scale_up",
            "ep_scope": "scale_up",
        },
        "interconnect": {
            "scale_up_latency": 4e-7,
            "scale_up_bandwidth": 450.0,
            "scale_up_energy": 1.3,
            "scale_out_latency": 2e-6,
            "scale_out_bandwidth": 50,
            "scale_out_energy": 1.3,
        },
        "kv_cache": {"max_context_length": 524288, "block_size": 256},
    }
}


def build_system_yaml(
    out_dir: Path,
    dram_yaml: Path,
    fixed_hardware: Dict[str, Any],
    core_num: int = 16,
) -> Path:
    """Write chip.yaml + system.yaml into out_dir, return system.yaml path."""
    out_dir.mkdir(parents=True, exist_ok=True)

    chip = json.loads(json.dumps(_CHIP_TEMPLATE))  # deep copy
    arch = chip["architecture"]
    arch["core_num"] = int(core_num)
    arch["frequency"] = float(fixed_hardware["frequency"])
    arch["core"]["matrix"]["mac_num"] = int(fixed_hardware["mac_num"])
    arch["core"]["vector"]["vec_num"] = int(fixed_hardware["vec_num"])
    arch["core"]["buffer"]["buffer_size"] = int(fixed_hardware["buffer_size"])
    arch["core"]["buffer"]["read_bw"] = float(fixed_hardware["buffer_read_bw"])
    arch["core"]["buffer"]["write_bw"] = float(fixed_hardware["buffer_write_bw"])
    arch["dram"]["config_path"] = str(dram_yaml)

    chip_path = out_dir / "chip.yaml"
    with chip_path.open("w") as f:
        yaml.safe_dump(chip, f, sort_keys=False)

    system = json.loads(json.dumps(_SYSTEM_TEMPLATE))
    system["system"]["chip"]["config_path"] = str(chip_path.resolve())

    system_path = out_dir / "system.yaml"
    with system_path.open("w") as f:
        yaml.safe_dump(system, f, sort_keys=False)
    return system_path


def run_one(
    bw_gbps: int,
    channel: int,
    density_gb: int,
    run_cfg: Dict[str, Any],
    tag: str,
) -> Dict[str, Any]:
    """Run one ATLAS inference. Returns the metrics dict (raises on failure)."""
    dram_yaml = dramcfg.write_cfg(channel, density_gb)

    out_dir = ARTIFACT_DIR / tag
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    system_yaml = build_system_yaml(
        out_dir, dram_yaml, run_cfg["fixed_hardware"], run_cfg["core_num"]
    )

    spec_path = out_dir / "spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "system_yaml": str(system_yaml),
                "model_preset": run_cfg["model_preset"],
                "batch_size": run_cfg["batch_size"],
                "context_length": run_cfg["context_length"],
                "element_size": run_cfg["element_size"],
            }
        )
    )

    result_path = out_dir / "metrics.json"
    artifacts = out_dir / "atlas_artifacts"
    artifacts.mkdir(exist_ok=True)

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{ATLAS_LIB}:" + env.get("LD_LIBRARY_PATH", "")
    env["PYTHONPATH"] = f"{ATLAS_LIB}:{ATLAS}:" + env.get("PYTHONPATH", "")
    env["AGENTDSE_TILING_WORKERS"] = str(run_cfg.get("num_workers", 4))

    log_path = out_dir / "atlas.log"
    # NOTE: subprocess.run gives each point its OWN OS process. This is REQUIRED,
    # not an optimization: booksim's rand() is only seeded inside noc_main(), so
    # running two points in one process consumes shared RNG state and shifts
    # e2e_cycles by ~0.4% (measured: 9642 -> 9683). Do NOT switch to in-process calls.
    with log_path.open("w") as log:
        proc = subprocess.run(
            [sys.executable, str(WORKER), str(spec_path), str(artifacts), str(result_path)],
            stdout=log,
            stderr=log,
            env=env,
            timeout=int(run_cfg.get("timeout_s", 900)),
        )

    if proc.returncode != 0 or not result_path.exists():
        raise RuntimeError(f"ATLAS failed (rc={proc.returncode}); see {log_path}")

    payload = json.loads(result_path.read_text())
    payload["bw_gbps"] = bw_gbps
    payload["channel"] = channel
    payload["density_gb"] = density_gb
    payload["run_dir"] = str(out_dir)
    return payload
```

### 3.5 创建 `agentdse/scripts/02_run_one.py`

```python
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentdse.paths import CONFIG_DIR, LOG_DIR
from agentdse.sim import run_one


def main() -> int:
    run_cfg = json.loads((CONFIG_DIR / "run.json").read_text())

    # 第一次跑就用 1 TB/s（16 channel x 4 Gb），和实测基线对齐
    metrics = run_one(
        bw_gbps=1024,
        channel=16,
        density_gb=4,
        run_cfg=run_cfg,
        tag="one_1024GBps",
    )

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = LOG_DIR / "step3_one_point.json"
    out.write_text(json.dumps(metrics, indent=2))

    print(json.dumps({k: metrics[k] for k in (
        "bw_gbps", "latency_s", "energy_j", "matrix_util", "dram_util",
        "dram_stall_cycles", "dram_energy_j", "elapsed_s")}, indent=2))
    print(f"written -> {out}")

    # 期望（实测）：latency≈3.34e-4, energy≈0.0402, dram_util≈0.688
    assert metrics["latency_s"] > 0
    assert metrics["energy_j"] > 0
    print("STEP3_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### 3.6 创建 `agentdse/__init__.py` 和 `agentdse/agentdse/__init__.py`

两个都空文件即可：

```bash
touch agentdse/__init__.py agentdse/agentdse/__init__.py
```

### 3.7 验证

```bash
cd /data2/nirunheng/Project/Beacon
python3 agentdse/scripts/02_run_one.py
```

约 1 分钟。必须打印 `STEP3_OK`，并且数值接近：
`latency_s≈0.000334`、`energy_j≈0.0402`、`matrix_util≈0.199`、`dram_util≈0.688`。

**如果输出的 `elapsed_s` 超过 300**，把 `run.json` 的 `num_workers` 提到 8 再试。
**如果 `atlas.log` 里有 `unsupported operator type`**，说明 Step 2.1 没生效。

---

## Step 4：造种子数据

目的：在跑真实仿真之前，把 Agent 链路先打通。

> **注意：真实候选点只有 7 个，不是 20 个**（原因见 §0.5 结论 A）。所以这一步造的是 7 条“真值形状”的种子数据，用于验证工具链与决策逻辑。

### 4.1 创建 `agentdse/config/space.json`

```json
{
  "objective": "cost = (latency/base_latency) * (energy/base_energy) * (dram_area/base_dram_area)",
  "baseline_bw_gbps": 512,
  "baseline_channel": 8,
  "points": [
    {"channel": 2,   "density_gb": 16},
    {"channel": 4,   "density_gb": 16},
    {"channel": 8,   "density_gb": 8},
    {"channel": 16,  "density_gb": 4},
    {"channel": 32,  "density_gb": 2},
    {"channel": 64,  "density_gb": 1},
    {"channel": 128, "density_gb": 1}
  ],
  "initial_offsets": [0, 1, 2, 6]
}
```

`initial_offsets` 是**故意跳开**的下标：先跑第 0、1、2、6 个点（128 / 256 / 512 / 8192 GB/s），**跳过 1024 和 2048**，也就是跳过了最优所在区间，让 Agent 有真正的决策空间。

> `baseline_bw_gbps` 必须是种子点之一（这里取 512，即第 2 个点），否则归一化基准缺失，`05_full_loop.py` 阶段 B 会警告。

实测代价曲线（cost 已含面积项，以 ch=16 为 1.0）：

| channel | 带宽 | latency s | energy J | dram_util | stall cycles | cost |
|---|---|---|---|---|---|---|
| 2 | 128 | 1.4613e-3 | 0.175438 | 0.9412 | 39106 | 2.2002 |
| 4 | 256 | 8.2554e-4 | 0.099141 | 0.8889 | 19358 | 1.4048 |
| 8 | 512 | 4.9238e-4 | 0.059162 | 0.7774 | 9002 | **1.0000**（baseline） |
| 16 | 1024 | 3.3414e-4 | 0.040174 | 0.6882 | 4178 | **0.9216**（最优） |
| 32 | 2048 | 2.6051e-4 | 0.031338 | 0.5672 | 1858 | 1.1210 |
| 64 | 4096 | 2.2950e-4 | 0.027617 | 0.5107 | 932 | 1.7406 |
| 128 | 8192 | 2.1926e-4 | 0.026388 | 0.4881 | 620 | 3.1779 |

**最优在 ch=16（1024 GB/s）**。`initial_offsets` 跳过 1024 和 2048，Agent 需要从 128→256→512 的下降趋势和 8192 的上升值，把 1024 这段找回来。

> 7 个点全是我这台机器上真跑出来的，逐位可复现。ch=2 容量只有 32 Gb（4 GB），也能跑通——实测峰值地址只有 0.16 GiB。

### 4.2 创建 `agentdse/scripts/03_make_seed.py`

```python
"""Write the 7 measured points as seed trials.

Numbers are from real ATLAS e2e runs (llama3_8b, batch=8, ctx=4096,
core_num=16, HBDRAM_500Mbps). cost includes the DRAM area term
A/A0 = channel/8, so the optimum is interior (channel=16).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentdse.paths import CONFIG_DIR, DATA_DIR

# 全部 MEASURED，独立进程跑出来逐位可复现
# channel, density_gb, latency_s, energy_j, matrix_util, dram_util, dram_stall, e2e_cycles
MEASURED = [
    (2,   16, 1.461344e-3, 0.1754376163328, 0.0428, 0.9412, 39106, 44867),
    (4,   16, 8.25536e-4,  0.0991406563328, 0.0768, 0.8889, 19358, 24998),
    (8,    8, 4.92384e-4,  0.0591624163328, 0.1316, 0.7774,  9002, 14587),
    (16,   4, 3.34144e-4,  0.0401736163328, 0.1991, 0.6882,  4178,  9642),
    (32,   2, 2.60512e-4,  0.0313377763328, 0.2615, 0.5672,  1858,  7341),
    (64,   1, 2.29504e-4,  0.0276168163328, 0.3013, 0.5107,   932,  6372),
    (128,  1, 2.19264e-4,  0.0263880163328, 0.3173, 0.4881,   620,  6052),
]


def cost_of(row, base, base_area):
    return (
        (row["latency_s"] / base["latency_s"])
        * (row["energy_j"] / base["energy_j"])
        * (row["channel"] / base_area)
    )


def main() -> int:
    space = json.loads((CONFIG_DIR / "space.json").read_text())
    out_dir = DATA_DIR / "seed"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for ch, den, lat, eng, mutil, dutil, stall, cycles in MEASURED:
        rows.append({
            "channel": ch, "density_gb": den, "bw_gbps": ch * 64,
            "latency_s": lat, "energy_j": eng,
            "matrix_util": mutil, "dram_util": dutil,
            "dram_stall_cycles": stall, "e2e_cycles": cycles,
            "status": "ok", "run_dir": "SEED",
        })

    base = next(r for r in rows if r["bw_gbps"] == space["baseline_bw_gbps"])
    base_area = space["baseline_channel"]
    for r in rows:
        r["cost"] = round(cost_of(r, base, base_area), 6)
        p = out_dir / f"trial_bw{r['bw_gbps']:05d}.json"
        p.write_text(json.dumps(r, indent=2))
        print(f"{p.name:26s} channel={r['channel']:>3} cost={r['cost']:.4f}")

    print(f"wrote {len(rows)} seed trials -> {out_dir}")
    print("STEP4_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### 4.3 验证

```bash
python3 agentdse/scripts/03_make_seed.py
cat agentdse/data/seed/trial_bw01024.json
```

必须看到 7 行、U 形曲线，且**以 ch=8 / 512 GB/s 为 1.0**：

```
trial_bw00128.json   channel=  2   cost=2.2002
trial_bw00256.json   channel=  4   cost=1.4048
trial_bw00512.json   channel=  8   cost=1.0000   <- baseline
trial_bw01024.json   channel= 16   cost=0.9216   <- 最低点
trial_bw02048.json   channel= 32   cost=1.1210
trial_bw04096.json   channel= 64   cost=1.7406
trial_bw08192.json   channel=128   cost=3.1779
STEP4_OK
```

同时 `trial_bw01024.json` 里应有 `matrix_util=0.1991`、`dram_util=0.6882`、
`dram_stall_cycles=4178`、`e2e_cycles=9642` —— 这四个字段 Step 8 的明细表要用。

---

## Step 5：Agent 读 JSON（工具层）

### 5.1 创建 `agentdse/agentdse/store.py`

sqlite 两张表。**这是唯一的"记忆"**，不要依赖 RAG。

```python
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from .paths import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS trials (
    fingerprint TEXT PRIMARY KEY,
    bw_gbps     INTEGER NOT NULL,
    channel     INTEGER NOT NULL,
    density_gb  INTEGER NOT NULL,
    status      TEXT NOT NULL,
    cost        REAL,
    latency_s   REAL,
    energy_j    REAL,
    record      TEXT NOT NULL,
    created_at  TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS decisions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    round       INTEGER NOT NULL,
    chosen_bw   INTEGER NOT NULL,
    reason      TEXT,
    evidence    TEXT,
    created_at  TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS best (
    id          INTEGER PRIMARY KEY CHECK (id = 1),
    bw_gbps     INTEGER NOT NULL,
    cost        REAL NOT NULL,
    updated_at  TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


class Store:
    def __init__(self, path: Path = DB_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ---- trials ----
    def put_trial(self, record: Dict[str, Any]) -> bool:
        """Insert. Returns True if new, False if fingerprint already known."""
        fp = f"bw{record['bw_gbps']}"
        try:
            self.conn.execute(
                "INSERT INTO trials (fingerprint,bw_gbps,channel,density_gb,status,cost,"
                "latency_s,energy_j,record) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    fp, record["bw_gbps"], record["channel"], record["density_gb"],
                    record.get("status", "ok"), record.get("cost"),
                    record.get("latency_s"), record.get("energy_j"),
                    json.dumps(record),
                ),
            )
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def all_trials(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT record FROM trials ORDER BY bw_gbps"
        ).fetchall()
        return [json.loads(r["record"]) for r in rows]

    def ok_trials(self) -> List[Dict[str, Any]]:
        return [r for r in self.all_trials() if r.get("status") == "ok"]

    def has_bw(self, bw: int) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM trials WHERE bw_gbps=?", (bw,)
        ).fetchone()
        return row is not None

    # ---- decisions ----
    def put_decision(self, rnd: int, bw: int, reason: str, evidence: str) -> None:
        self.conn.execute(
            "INSERT INTO decisions (round,chosen_bw,reason,evidence) VALUES (?,?,?,?)",
            (rnd, bw, reason, evidence),
        )
        self.conn.commit()

    def decisions(self) -> List[Dict[str, Any]]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM decisions ORDER BY id").fetchall()]

    # ---- best ----
    def update_best(self, bw: int, cost: float) -> str:
        """Returns 'inserted' | 'updated' | 'kept'."""
        row = self.conn.execute("SELECT bw_gbps,cost FROM best WHERE id=1").fetchone()
        if row is None:
            self.conn.execute("INSERT INTO best (id,bw_gbps,cost) VALUES (1,?,?)", (bw, cost))
            self.conn.commit()
            return "inserted"
        if cost < row["cost"]:
            self.conn.execute("UPDATE best SET bw_gbps=?,cost=? WHERE id=1", (bw, cost))
            self.conn.commit()
            return "updated"
        return "kept"

    def best(self) -> Optional[Dict[str, Any]]:
        row = self.conn.execute("SELECT * FROM best WHERE id=1").fetchone()
        return dict(row) if row else None
```

### 5.2 创建 `agentdse/agentdse/tools.py`

四个工具，**全部返回可直接塞进 prompt 的 dict**。

```python
from __future__ import annotations

import json
from typing import Any, Dict, List

from .store import Store

# ---- 工具 1：观察当前排行榜 -------------------------------------------


def tool_lookup(store: Store, top_k: int = 20) -> Dict[str, Any]:
    rows = store.ok_trials()
    ranked = sorted(rows, key=lambda r: r["cost"])[:top_k]
    return {
        "tool": "lookup_trials",
        "n_total": len(store.all_trials()),
        "n_ok": len(rows),
        "best": store.best(),
        "ranked": [_compact(r) for r in ranked],
    }


# ---- 工具 2：看某个带宽点的完整观测 -----------------------------------


def tool_inspect(store: Store, bw: int) -> Dict[str, Any]:
    for r in store.all_trials():
        if r["bw_gbps"] == bw:
            return {"tool": "inspect_point", "found": True, "point": r}
    return {"tool": "inspect_point", "found": False, "bw_gbps": bw,
            "note": "this bandwidth has not been simulated yet"}


# ---- 工具 3：看曲线趋势（相邻点的一阶差分） ---------------------------


def tool_trend(store: Store) -> Dict[str, Any]:
    rows = sorted(store.ok_trials(), key=lambda r: r["bw_gbps"])
    steps: List[Dict[str, Any]] = []
    for a, b in zip(rows, rows[1:]):
        d_cost = b["cost"] - a["cost"]
        steps.append({
            "from_bw": a["bw_gbps"],
            "to_bw": b["bw_gbps"],
            "d_cost": round(d_cost, 4),
            "d_cost_pct": round(100.0 * d_cost / a["cost"], 2) if a["cost"] else None,
            "d_latency_pct": round(100.0 * (b["latency_s"] - a["latency_s"]) / a["latency_s"], 2),
            "d_energy_pct": round(100.0 * (b["energy_j"] - a["energy_j"]) / a["energy_j"], 2),
            "direction": "better" if d_cost < 0 else ("worse" if d_cost > 0 else "flat"),
        })
    return {"tool": "trend", "n_points": len(rows), "steps": steps}


# ---- 工具 4：校验一个提案是否合法 -------------------------------------


def tool_validate(store: Store, bw: int, space_points: List[Dict[str, int]]) -> Dict[str, Any]:
    legal = {p["channel"] * 64 for p in space_points}
    errors: List[str] = []
    if bw not in legal:
        errors.append(
            f"bw={bw} GB/s is not in the design space; legal values are {sorted(legal)}"
        )
    if store.has_bw(bw):
        errors.append(f"bw={bw} GB/s has already been simulated; pick an untried value")
    return {
        "tool": "validate_proposal",
        "ok": not errors,
        "errors": errors,
        "legal_values": sorted(legal),
    }


def _compact(r: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "bw_gbps": r["bw_gbps"],
        "cost": r["cost"],
        "latency_s": r["latency_s"],
        "energy_j": r["energy_j"],
        "dram_util": r.get("dram_util"),
        "matrix_util": r.get("matrix_util"),
        "dram_stall_cycles": r.get("dram_stall_cycles"),
    }


TOOL_SPECS = [
    {
        "name": "lookup_trials",
        "description": "List all simulated bandwidth points, ranked by cost (lower is better).",
        "arguments": {"top_k": "optional int, default 20"},
    },
    {
        "name": "trend",
        "description": "Show first differences between adjacent bandwidth points: is cost still improving?",
        "arguments": {},
    },
    {
        "name": "inspect_point",
        "description": "Return the full record of one already-simulated bandwidth value.",
        "arguments": {"bw_gbps": "int"},
    },
    {
        "name": "validate_proposal",
        "description": "Check whether a proposed bandwidth is legal and not yet simulated. Call before proposing.",
        "arguments": {"bw_gbps": "int"},
    },
]


def dispatch(store: Store, name: str, args: Dict[str, Any],
             space_points: List[Dict[str, int]]) -> Dict[str, Any]:
    if name == "lookup_trials":
        return tool_lookup(store, int(args.get("top_k", 20)))
    if name == "trend":
        return tool_trend(store)
    if name == "inspect_point":
        return tool_inspect(store, int(args["bw_gbps"]))
    if name == "validate_proposal":
        return tool_validate(store, int(args["bw_gbps"]), space_points)
    raise ValueError(f"unknown tool: {name}")
```

### 5.3 验证

```bash
cd /data2/nirunheng/Project/Beacon
python3 - <<'PY'
import json, sys
sys.path.insert(0, "agentdse")
from agentdse.store import Store
from agentdse import tools
from agentdse.paths import DATA_DIR, CONFIG_DIR

s = Store(DATA_DIR / "test.db")
for p in sorted((DATA_DIR / "seed").glob("*.json")):
    s.put_trial(json.loads(p.read_text()))

space = json.loads((CONFIG_DIR / "space.json").read_text())["points"]
for r in s.ok_trials():
    s.update_best(r["bw_gbps"], r["cost"])

print(json.dumps(tools.tool_lookup(s, 3), indent=2))
print(json.dumps(tools.tool_trend(s)["steps"][-3:], indent=2))
print(json.dumps(tools.tool_validate(s, 1536, space), indent=2))
print("STEP5_OK")
PY
```

必须看到 `best` 是 `{"bw_gbps": 1024, "cost": 0.9216}`，并且 `validate_proposal` 对 1536 报错（1536 不在合法集合里，因为 channel 必须是 2 的幂），最后打印 `STEP5_OK`。

---

## Step 6：Agent 决策

### 6.1 创建 `agentdse/agentdse/prompts.py`

```python
SYSTEM_DIAGNOSE = """\
You are a hardware design-space-exploration analyst.
You read simulator results and identify where the bottleneck is and whether
adding DRAM bandwidth is still worthwhile.

Rules:
- Only use numbers that appear in the data provided. Never invent values.
- `cost` is (latency/base_latency) * (energy/base_energy) * (dram_area/base_area); LOWER IS BETTER.
- `dram_area` scales with channel count, so `cost` has an INTERIOR minimum:
  adding bandwidth eventually costs more area than the latency you buy back.
- `dram_util` near 1.0 means DRAM is saturated. `matrix_util` low while
  `dram_util` high means the design is memory-bound.
- `dram_stall_cycles` is how long compute units waited on DRAM. If it is
  already near zero, more bandwidth cannot help.
- Output strict JSON, no prose, no markdown fences.
"""

DIAGNOSE_USER = """\
## Design space
{space}

## All simulated points (sorted by bandwidth)
{trials}

## Adjacent-point trend
{trend}

## Current best
{best}

Produce JSON:
{{
  "bottleneck": "memory_bound | compute_bound | balanced | unknown",
  "still_improving": true or false,
  "marginal_gain_pct": number,
  "evidence": ["<quote a concrete number from the data>", "..."],
  "summary": "<one sentence a human can read>"
}}

Field semantics (get these right, they are easy to misread):
- `marginal_gain_pct` = the percent by which the BEST cost could STILL drop if
  you found a better point. It is never negative. It is NOT "how much cost rose
  between the cheapest and the most expensive point". If you cannot estimate it,
  return 0.
- `still_improving` = whether cost is still going DOWN as bandwidth grows inside
  the region you have already sampled. Crossing the interior minimum flips this
  to false, which is the normal expected outcome and NOT itself a reason to stop.
"""


SYSTEM_PROPOSE = """\
You choose the NEXT bandwidth value to simulate for a hardware DSE loop.
You are given the diagnosis, the full history, the legal design space, and
which values are still untried.

Rules:
- The chosen value MUST be in the legal set and MUST NOT have been simulated.
- Prefer interpolating inside a gap where the trend suggests the optimum lies.
- If the diagnosis says the curve has flattened, choose a nearby point to
  confirm the knee rather than jumping far away.
- Output strict JSON, no prose, no markdown fences.
"""

PROPOSE_USER = """\
## Diagnosis
{diagnosis}

## Already simulated (bandwidth GB/s -> cost)
{history}

## Untried legal values (GB/s)
{untried}

## Best so far
{best}

Produce JSON:
{{
  "bw_gbps": <int, must be one of the untried values>,
  "reason": "<why this value, referencing the numbers>",
  "expected_direction": "lower_cost | higher_cost | uncertain",
  "confidence": "low | medium | high"
}}
"""
```

### 6.2 创建 `agentdse/agentdse/agent.py`

```python
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from . import tools
from .llm import LLM
from .prompts import (
    DIAGNOSE_USER, PROPOSE_USER, SYSTEM_DIAGNOSE, SYSTEM_PROPOSE,
)
from .store import Store


class Agent:
    def __init__(self, llm: LLM, store: Store, space: Dict[str, Any]) -> None:
        self.llm = llm
        self.store = store
        self.space = space
        self.legal = sorted(p["channel"] * 64 for p in space["points"])

    # ---------------- 内部 ----------------
    def _untried(self) -> List[int]:
        return [bw for bw in self.legal if not self.store.has_bw(bw)]

    def _space_brief(self) -> str:
        return (
            f"One tunable parameter: DRAM bandwidth, {len(self.legal)} discrete values "
            f"from {self.legal[0]} to {self.legal[-1]} GB/s.\n"
            f"All other hardware fields are fixed: {self.space.get('fixed')}\n"
            f"Legal set: {self.legal}"
        )

    def _history_brief(self) -> str:
        rows = sorted(self.store.ok_trials(), key=lambda r: r["bw_gbps"])
        if not rows:
            return "(empty)"
        return "\n".join(
            f"  bw={r['bw_gbps']:>5}  cost={r['cost']:.4f}  latency={r['latency_s']:.6e}"
            f"  energy={r['energy_j']:.6e}  dram_util={r.get('dram_util')}"
            f"  dram_stall={r.get('dram_stall_cycles')}"
            for r in rows
        )

    # ---------------- 第一段：诊断 ----------------
    def diagnose(self) -> Dict[str, Any]:
        trend = tools.tool_trend(self.store)
        user = DIAGNOSE_USER.format(
            space=self._space_brief(),
            trials=self._history_brief(),
            trend=json.dumps(trend["steps"], indent=2),
            best=json.dumps(self.store.best(), indent=2),
        )
        result = self.llm.chat_json(SYSTEM_DIAGNOSE, user)
        if not result.get("evidence"):
            raise ValueError(f"diagnosis has no evidence: {result}")
        return result

    # ---------------- 第二段：提案 ----------------
    def propose(self, diagnosis: Dict[str, Any], rnd: int) -> Dict[str, Any]:
        untried = self._untried()
        if not untried:
            return {"stop": True, "reason": "every legal bandwidth value has been simulated"}

        user = PROPOSE_USER.format(
            diagnosis=json.dumps(diagnosis, indent=2),
            history=self._history_brief(),
            untried=untried,
            best=json.dumps(self.store.best(), indent=2),
        )

        last_error: Optional[str] = None
        for attempt in range(3):
            if last_error:
                user_with_error = user + f"\n\n## Your previous answer was rejected\n{last_error}\n"
            else:
                user_with_error = user

            result = self.llm.chat_json(SYSTEM_PROPOSE, user_with_error)

            if result.get("stop") is True:
                return {"stop": True, "reason": result.get("reason", "agent chose to stop")}

            bw = result.get("bw_gbps")
            try:
                bw = int(bw)
            except (TypeError, ValueError):
                last_error = f"bw_gbps must be an integer, got {bw!r}"
                continue

            check = tools.tool_validate(self.store, bw, self.space["points"])
            if not check["ok"]:
                last_error = "; ".join(check["errors"])
                continue

            self.store.put_decision(
                rnd, bw,
                reason=str(result.get("reason", "")),
                evidence=json.dumps(diagnosis.get("evidence", [])),
            )
            return {"stop": False, "bw_gbps": bw, **result}

        raise ValueError(f"agent failed to produce a legal proposal in 3 attempts: {last_error}")
```

### 6.3 创建 `agentdse/scripts/04_ask_agent.py`

**只读，不跑仿真。** 用来验证 Step 4 的种子数据 + Step 5 的工具 + Step 6 的 prompt 真的通了。

```python
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentdse.agent import Agent
from agentdse.llm import LLM
from agentdse.paths import CONFIG_DIR, DATA_DIR, KEY_JSON, LOG_DIR
from agentdse.store import Store


def main() -> int:
    db = DATA_DIR / "agent_test.db"
    if db.exists():
        db.unlink()
    store = Store(db)

    space = json.loads((CONFIG_DIR / "space.json").read_text())
    # 只灌 initial_offsets 指定的点。灌全部 7 点会让 untried 为空，
    # Agent 只能返回 stop=True，就验证不了"能否挑出一个未试过的点"。
    legal = sorted(p["channel"] * 64 for p in space["points"])
    want = {legal[i] for i in space["initial_offsets"]}
    for p in sorted((DATA_DIR / "seed").glob("*.json")):
        rec = json.loads(p.read_text())
        if rec["bw_gbps"] in want:
            store.put_trial(rec)
    for r in store.ok_trials():
        store.update_best(r["bw_gbps"], r["cost"])

    print(f"loaded {len(store.all_trials())} seed trials "
          f"(initial_offsets only: {sorted(want)})")
    print(f"best so far: {store.best()}")

    agent = Agent(LLM.from_file(KEY_JSON), store, space)

    diagnosis = agent.diagnose()
    print("\n=== diagnosis ===")
    print(json.dumps(diagnosis, indent=2, ensure_ascii=False))

    choice = agent.propose(diagnosis, rnd=1)
    print("\n=== next proposal ===")
    print(json.dumps(choice, indent=2, ensure_ascii=False))

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "step6_decision.json").write_text(
        json.dumps({"diagnosis": diagnosis, "proposal": choice}, indent=2, ensure_ascii=False)
    )

    if choice.get("stop"):
        print("\nagent stopped early")
    else:
        assert choice["bw_gbps"] in agent._untried(), "proposal must be untried"
        print(f"\nchosen bw = {choice['bw_gbps']} GB/s (untried)")
    print("STEP6_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### 6.4 验证

```bash
python3 agentdse/scripts/04_ask_agent.py
```

必须看到：
- `loaded 4 seed trials (initial_offsets only: [128, 256, 512, 8192])`
- `=== diagnosis ===` 里 `bottleneck` 是 `memory_bound`，`evidence` 非空
- `=== next proposal ===` 里 `bw_gbps` 必须是 `1024` 或 `2048`（这两个是真正的最优区，且都未试过）
- `chosen bw = ... GB/s (untried)` 和 `STEP6_OK`

**注意 diagnosis 里的 `still_improving` 和 `marginal_gain_pct` 是 LLM 自报的，实测不可靠**
（它会把 "cost 上涨 217%" 误当成边际收益，还给出 `still_improving=false`）。
这两个字段只作为给 LLM 自己看的上下文，**真正的早停判据在 Step 7 由代码算**。

**如果 diagnosis 说 `evidence` 为空** → prompt 没生效，检查 `SYSTEM_DIAGNOSE` 是否被截断。
**如果 proposal 反复被拒** → 打印 `last_error`，通常是 LLM 选了已试过的值，这正说明 `validate_proposal` 在工作。

---

## Step 7：Agent 调仿真 + 更新 best

### 7.1 创建 `agentdse/scripts/05_full_loop.py`

```python
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentdse import dramcfg
from agentdse.agent import Agent
from agentdse.llm import LLM
from agentdse.paths import ARTIFACT_DIR, CONFIG_DIR, DATA_DIR, KEY_JSON, LOG_DIR
from agentdse.sim import run_one
from agentdse.store import Store

MAX_ITERS = 15
MARGINAL_THRESHOLD_PCT = 3.0

BASE_CHANNEL = 8           # area normalisation reference; must be in space.json as baseline_channel


def _cost(row, base, base_ch=BASE_CHANNEL):
    """(L/L0) * (E/E0) * (A/A0).  Area scales linearly with channel count,
    which is what creates an INTERIOR optimum (see manual §0.5 结论 C)."""
    return (
        (row["latency_s"] / base["latency_s"])
        * (row["energy_j"] / base["energy_j"])
        * (row["channel"] / base_ch)
    )


def channel_of(space, bw):
    for p in space["points"]:
        if p["channel"] * 64 == bw:
            return p
    raise KeyError(f"bw {bw} not in space")


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    trace_path = LOG_DIR / "loop_trace.jsonl"
    if trace_path.exists():
        trace_path.unlink()

    run_cfg = json.loads((CONFIG_DIR / "run.json").read_text())
    space = json.loads((CONFIG_DIR / "space.json").read_text())
    legal = sorted(p["channel"] * 64 for p in space["points"])

    store = Store(DATA_DIR / "trials.db")

    # 稳健性：允许 data/dram_cfgs 与 artifacts 在多次运行间复用，
    # 但 trials.db 会保留历史。若你想从零重跑，先删 data/trials.db。
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] start  db={DATA_DIR / 'trials.db'}  "
          f"existing trials={len(store.ok_trials())}")

    # ---- 阶段 A：先补齐 initial_offsets 指定的种子点 ----
    if not store.ok_trials():
        for idx in space["initial_offsets"]:
            bw = legal[idx]
            if store.has_bw(bw):
                continue
            p = channel_of(space, bw)
            print(f"[init] simulating bw={bw} GB/s ...", flush=True)
            t0 = time.time()
            try:
                rec = run_one(bw, p["channel"], p["density_gb"], run_cfg, tag=f"bw{bw}")
                rec["status"] = "ok"
            except Exception as exc:
                rec = {"bw_gbps": bw, "channel": p["channel"],
                       "density_gb": p["density_gb"], "status": "sim_error",
                       "error": str(exc), "elapsed_s": round(time.time() - t0, 1)}
            store.put_trial(rec)
            # NOTE: no update_best() here — cost is not normalised yet (baseline
            # may not have been simulated). 阶段 B recomputes cost and seeds best.
            print(f"       -> {rec['status']} "
                  f"({rec.get('elapsed_s')}s) best={store.best()}", flush=True)

    # ---- 阶段 B：算 baseline cost ----
    rows = store.ok_trials()
    if not rows:
        raise SystemExit("no successful trials; cannot compute baseline")
    base = next((r for r in rows if r["bw_gbps"] == space["baseline_bw_gbps"]), None)
    if base is None:
        base = min(rows, key=lambda r: abs(r["bw_gbps"] - space["baseline_bw_gbps"]))
        print(f"warning: baseline {space['baseline_bw_gbps']} missing, using {base['bw_gbps']}")
    base_ch = int(space.get("baseline_channel", BASE_CHANNEL))
    # put_trial() is INSERT-only (fingerprint PK), so cost must be written back
    # with an explicit UPDATE.  Do it in one pass over the raw records.
    for r in store.ok_trials():
        r["cost"] = round(_cost(r, base, base_ch), 6)
        store.conn.execute(
            "UPDATE trials SET cost=?, record=? WHERE fingerprint=?",
            (r["cost"], json.dumps(r), f"bw{r['bw_gbps']}"),
        )
    store.conn.commit()
    # best can only be seeded now, because cost was undefined before normalisation
    for r in store.ok_trials():
        store.update_best(r["bw_gbps"], r["cost"])

    print(f"baseline bw={base['bw_gbps']} cost=1.0")
    print(f"best after baseline normalisation: {store.best()}")

    # ---- 阶段 C：Agent 迭代 ----
    agent = Agent(LLM.from_file(KEY_JSON), store, space)
    stop_reason = "reached iteration limit"
    last_best_cost = None

    for rnd in range(1, MAX_ITERS + 1):
        print(f"\n{'='*60}\n[round {rnd}]", flush=True)

        best_now = store.best()
        # 边际收益必须由代码算，不能信 LLM 自报的数字：
        # 实测 LLM 会把“cost 上涨 217%”当成 marginal_gain_pct 返回，导致误停。
        marginal = None
        if last_best_cost is not None and best_now is not None:
            marginal = round(
                100.0 * (last_best_cost - best_now["cost"]) / last_best_cost, 3
            )
        last_best_cost = best_now["cost"] if best_now else None

        if marginal is not None and marginal < MARGINAL_THRESHOLD_PCT:
            stop_reason = (
                f"early stop: marginal gain of best cost = {marginal}% "
                f"< threshold {MARGINAL_THRESHOLD_PCT}% "
                f"(best={best_now})"
            )
            print(stop_reason, flush=True)
            with trace_path.open("a") as f:
                f.write(json.dumps({
                    "round": rnd, "event": "early_stop",
                    "marginal_gain_pct": marginal,
                    "stop_reason": stop_reason,
                    "best": best_now,
                }, ensure_ascii=False) + "\n")
            break

        diagnosis = agent.diagnose()
        diagnosis["marginal_gain_pct_code"] = marginal
        print("diagnosis:", json.dumps(diagnosis, ensure_ascii=False), flush=True)

        choice = agent.propose(diagnosis, rnd)
        if choice.get("stop"):
            stop_reason = f"agent stopped: {choice.get('reason')}"
            print(stop_reason, flush=True)
            break

        bw = int(choice["bw_gbps"])
        p = channel_of(space, bw)
        print(f"agent chose bw={bw} GB/s :: {choice.get('reason')}", flush=True)

        t0 = time.time()
        try:
            rec = run_one(bw, p["channel"], p["density_gb"], run_cfg, tag=f"bw{bw}")
            rec["status"] = "ok"
        except Exception as exc:
            rec = {"bw_gbps": bw, "channel": p["channel"], "density_gb": p["density_gb"],
                   "status": "sim_error", "error": str(exc),
                   "elapsed_s": round(time.time() - t0, 1)}

        updated = "none"
        if rec["status"] == "ok":
            rec["cost"] = round(_cost(rec, base, base_ch), 6)
            store.put_trial(rec)
            updated = store.update_best(bw, rec["cost"])

        best = store.best()
        if rec["status"] == "ok":
            print(
                f"  round {rnd:<2} bw={bw:>5} GB/s  L={rec['latency_s']:.6e}s  "
                f"E={rec['energy_j']:.6f}J  cost={rec['cost']:.4f}  "
                f"dram_util={rec.get('dram_util', 0):.4f}  "
                f"stall={rec.get('dram_stall_cycles')}  "
                f"{rec.get('elapsed_s')}s  status=ok  best={best['bw_gbps']}/"
                f"{best['cost']:.4f} ({updated})",
                flush=True,
            )
        else:
            print(
                f"  round {rnd:<2} bw={bw:>5} GB/s  status={rec['status']}  "
                f"{rec.get('elapsed_s')}s  best={best['bw_gbps']}/{best['cost']:.4f} "
                f"({updated})  error={rec.get('error', '')[:120]}",
                flush=True,
            )

        with trace_path.open("a") as f:
            f.write(json.dumps({
                "round": rnd,
                "diagnosis": diagnosis,
                "proposal": choice,
                "result": {k: rec.get(k) for k in (
                    "bw_gbps", "status", "cost", "latency_s", "energy_j",
                    "dram_util", "matrix_util", "dram_stall_cycles", "elapsed_s")},
                "best_after": best,
                "best_update": updated,
            }, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}\nstop_reason: {stop_reason}")
    print(f"simulated {len(store.ok_trials())} points out of {len(legal)} legal values")
    print(f"final best: {store.best()}")
    print(f"trace -> {trace_path}")
    print("STEP7_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### 7.2 验证（不跑仿真，只验库逻辑）

在跑真仿真前，先确认 store 的去重与 best 维护没问题：

```bash
# 临时验证：把 data/seed 复制成 trials.db，再单独跑 Agent 迭代
python3 - <<'PY'
import json, sys
sys.path.insert(0, "agentdse")
from agentdse.store import Store
from agentdse.paths import DATA_DIR
s = Store(DATA_DIR / "loop_dryrun.db")
for p in sorted((DATA_DIR / "seed").glob("*.json")):
    s.put_trial(json.loads(p.read_text()))
for r in s.ok_trials():
    s.update_best(r["bw_gbps"], r["cost"])
print("best:", s.best(), "n:", len(s.ok_trials()))
print("STEP7_DRYRUN_OK")
PY
```

### 7.3 验证（真仿真）

```bash
cd /data2/nirunheng/Project/Beacon
python3 agentdse/scripts/05_full_loop.py 2>&1 | tee agentdse/logs/step7_console.log
```

预期（**以下是我真跑一遍的输出形状，可直接对照**）：
- 阶段 A 跑 4 个点（`initial_offsets=[0,1,2,6]`），实测 4×~45 s ≈ 3 分钟
- 阶段 C 跑 2 个点后早停，实测 2×~35 s ≈ 1 分钟
- 每轮打印 `diagnosis` / `agent chose bw=...` / 一行 `round N bw=... cost=... best=... (updated|kept)`
- 最后打印 `stop_reason` 和 `STEP7_OK`

真实运行结果（本机，逐位可复现）：

```
[init] 128 / 256 / 512 / 8192 四点  -> ok
baseline bw=512 cost=1.0
[round 1] agent chose 1024  -> cost=0.9216  best=1024/0.9216 (updated)
[round 2] agent chose 2048  -> cost=1.1210  best=1024/0.9216 (kept)
[round 3] early stop: marginal gain of best cost = 0.0% < threshold 3.0%
simulated 6 points out of 7 legal values
```

注意 `marginal_gain_pct_code` 才是代码算的：
- round 1 是 `null`（只有一个 best，算不出边际收益）
- round 2 是 `7.837`（best 从 1.0 降到 0.9216）
- round 3 是 `0.0`（round 2 没刷新 best）→ 触发早停

**LLM 自己报的 `marginal_gain_pct` 不可信**（实测它会把 "cost 上涨 217%" 当成边际收益返回）。
所以早停判据用 `marginal_gain_pct_code`，不是 LLM 那个字段。

**验收三个硬指标：**

```bash
# 1) 是否真的只跑了部分点
sqlite3 agentdse/data/trials.db "SELECT count(*) FROM trials;"
# → 应该 < 7（例如 5~6，说明 Agent 真的只补了需要补的点）

# 2) 最优是否被正确记录
sqlite3 agentdse/data/trials.db "SELECT * FROM best;"

# 3) 是否每次都记录了决策
sqlite3 agentdse/data/trials.db "SELECT round,chosen_bw FROM decisions ORDER BY round;"
```

---

## Step 8：出图（汇报用）

创建 `agentdse/scripts/06_plot.py`：

```python
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agentdse.paths import DATA_DIR, LOG_DIR

DB = DATA_DIR / "trials.db"


def rows():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    out = []
    for r in conn.execute("SELECT * FROM trials WHERE status='ok' ORDER BY bw_gbps"):
        d = dict(r)
        # dram_util / dram_stall_cycles / matrix_util 只存在 record JSON 里，
        # 不在 trials 表的列中。不合并回来，明细表这两列会全是 0。
        if d.get("record"):
            try:
                extra = json.loads(d["record"])
            except (TypeError, ValueError):
                extra = {}
            for k, v in extra.items():
                d.setdefault(k, v)
        out.append(d)
    return out


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    data = rows()
    bw = [r["bw_gbps"] for r in data]
    cost = [r["cost"] for r in data]

    decisions = [json.loads(l) for l in (LOG_DIR / "loop_trace.jsonl").read_text().splitlines()]

    fig, axes = plt.subplots(1, 3, figsize=(19, 5))

    ax = axes[0]
    ax.plot(bw, cost, "o-", label="cost = (L/L0)*(E/E0)*(A/A0)")
    for d in decisions:
        res = d.get("result") or {}
        if res.get("bw_gbps") is None or res.get("cost") is None:
            continue
        ax.scatter(res["bw_gbps"], res["cost"], s=150,
                   marker="*", color="crimson", zorder=5)
    if decisions:
        ax.scatter([], [], s=150, marker="*", color="crimson", label="agent-chosen")
    best_i = min(range(len(cost)), key=lambda i: cost[i])
    ax.scatter([bw[best_i]], [cost[best_i]], s=220, marker="v",
               color="green", zorder=6, label=f"best = {bw[best_i]} GB/s")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("DRAM bandwidth (GB/s)")
    ax.set_ylabel("cost (lower is better)")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("cost vs bandwidth (U-shaped, interior min)")

    ax = axes[1]
    lat = [r["latency_s"] for r in data]
    eng = [r["energy_j"] for r in data]
    ax.plot(bw, [v / lat[0] for v in lat], "o-", label="latency / first")
    ax.plot(bw, [v / eng[0] for v in eng], "s-", label="energy / first")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("DRAM bandwidth (GB/s)")
    ax.set_ylabel("normalised to first point")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("latency and energy move together")

    ax = axes[2]
    # 三个量必须用同一个归一化基准（baseline 点 = cost 1.0 的那个），
    # 否则乘性关系看不出来：cost = latency_term * energy_term * area_term。
    i0 = next((i for i, c in enumerate(cost) if abs(c - 1.0) < 1e-6), 0)
    lat_term = [r["latency_s"] / data[i0]["latency_s"] for r in data]
    eng_term = [r["energy_j"] / data[i0]["energy_j"] for r in data]
    area_term = [r["channel"] / data[i0]["channel"] for r in data]
    ax.plot(bw, cost, "o-", label="cost = Lterm*Eterm*Aterm")
    ax.plot(bw, lat_term, "--", label="latency term (L/L0)")
    ax.plot(bw, eng_term, "-.", label="energy term (E/E0)")
    ax.plot(bw, area_term, ":", label="area term (A/A0)")
    ax.axhline(1.0, color="grey", lw=0.8, alpha=0.6)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("DRAM bandwidth (GB/s)")
    ax.set_ylabel("normalised to baseline (512 GB/s)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("why the optimum is interior")

    out = LOG_DIR / "report.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

    # ---- table (the third deliverable) ----
    lines = [
        f"{'bw':>6} {'ch':>4} {'latency_s':>12} {'energy_J':>10} "
        f"{'cost':>8} {'dram_util':>10} {'stall':>7} {'status':>9}"
    ]
    for r in data:
        lines.append(
            f"{r['bw_gbps']:>6} {r['channel']:>4} {r['latency_s']:>12.6e} "
            f"{r['energy_j']:>10.5f} {r['cost']:>8.4f} "
            f"{(r.get('dram_util') or 0):>10.4f} "
            f"{(r.get('dram_stall_cycles') or 0):>7} {r['status']:>9}"
        )
    table = "\n".join(lines)
    (LOG_DIR / "report_table.txt").write_text(table + "\n")
    print(table)
    print(f"best point: {bw[best_i]} GB/s, cost={cost[best_i]:.4f}")
    print("STEP8_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

```bash
python3 agentdse/scripts/06_plot.py
```

预期（真实运行输出）：

```
wrote agentdse/logs/report.png
    bw   ch    latency_s   energy_J     cost  dram_util   stall    status
   128    2 1.461344e-03    0.17544   2.2002     0.9412   39106        ok
   256    4 8.255360e-04    0.09914   1.4048     0.8889   19358        ok
   512    8 4.923840e-04    0.05916   1.0000     0.7774    9002        ok
  1024   16 3.341440e-04    0.04017   0.9216     0.6882    4178        ok
  2048   32 2.605120e-04    0.03134   1.1210     0.5672    1858        ok
  8192  128 2.192640e-04    0.02639   3.1779     0.4881     620        ok
best point: 1024 GB/s, cost=0.9216
STEP8_OK
```

三张图各自回答一个问题：

| 图 | 内容 | 汇报时说什么 |
|---|---|---|
| 左（`cost vs bandwidth`） | U 形 cost 曲线 + 红星（Agent 逐轮选点）+ 绿三角（最终 best） | 7 个点里只跑了 6 个，Agent 自己把最优 1024 找回来了 |
| 中（`latency and energy move together`） | L 与 E 两条线几乎重合（都归一到首点） | 功率恒定 ≈120 W，所以 E≈120×L，单靠 L、E 无法形成内部极小 |
| 右（`why the optimum is interior`） | Lterm / Eterm / Aterm 与总 cost，同一基准归一 | 面积项线性上涨，与下降的 L/E 项交叉，交叉点就是 1024 |

**如果 `dram_util` / `stall` 列全是 0**：说明 `rows()` 没把 `record` JSON 合并回来，检查 Step 8 代码里那段 `json.loads(d["record"])`。

---

## 9. 检查清单（周五早上自查）

| # | 项 | 命令 / 判据 |
|---|---|---|
| 1 | API 通 | `python3 agentdse/scripts/01_check_llm.py` → `STEP1_OK` |
| 2 | 补丁生效 | `grep -c "str(SimulatorOperatorType" ATLAS-MICRO-2026-main/frontend/auto/*.py` → 全 0 |
| 3 | 单点仿真通 | `python3 agentdse/scripts/02_run_one.py` → `STEP3_OK`，latency≈3.34e-4 |
| 4 | 种子可读 | `python3 agentdse/scripts/03_make_seed.py` → 7 个文件，cost 曲线 U 形（以 512 为 1.0，最低在 1024 = 0.9216） |
| 5 | 工具可用 | Step 5.3 的片段 → `STEP5_OK` |
| 6 | 决策可读 | `python3 agentdse/scripts/04_ask_agent.py` → `STEP6_OK`，`evidence` 非空 |
| 7 | 完整循环 | `python3 agentdse/scripts/05_full_loop.py` → `STEP7_OK`，`trials < 7` |
| 8 | 最优被更新 | `sqlite3 agentdse/data/trials.db "SELECT * FROM best"` 非空 |
| 9 | Trace 完整 | `wc -l agentdse/logs/loop_trace.jsonl` ≥ 2 |
| 10 | 图存在 | `ls agentdse/logs/report.png` |
| 11 | 每点独立进程 | 同一配置连跑两次，`e2e_cycles` 必须完全相同（9642/9642/9642） |
| 12 | 无假点 | `space.json` 的 channel 全是 2 的幂；`dramcfg.make_cfg` 会拒绝非 2 的幂 |

---

## 10. 七个已知的坑（都已在代码里避开，但你要知道原因）

**① `core_num` 必须是偶数平方。**
`tiling_explorer_cloud.py:2286`：`core_array_size = int(math.sqrt(core_num))`，且网格边长必须是偶数。
合法值：4, 16, 64, 256。**用 16，不要改。**

**② `LD_LIBRARY_PATH` 必须在进程启动前设好。**
`libramulator.so` 找不到时 `atlasim` 导入失败。`sim.py` 里通过 `subprocess` 的 `env` 传，这是唯一可靠的方式。

**③ `multiprocessing.cpu_count` 的修补必须早于 `import frontend.auto.*`。**
`num_workers=int(0.8*cpu_count())` 是**默认参数**，在模块导入时求值一次。晚一行就没用。
`sim_worker.py` 里就是它，别往上/下移。

**④ `str(SimulatorOperatorType.X)` 必须换成 `.to_string()`。**
否则 C++ 端 `exit(-1)`，而父进程在等 Pool → 表现为"永久挂起"，极难排查。

**⑤ `cost` 的 baseline 必须存在。**
`space.json` 里 `baseline_bw_gbps=512`，它必须在 `initial_offsets` 里被跑到，否则 `05_full_loop.py` 阶段 B 会打 warning 并用最近点代替。

**⑥ `channel` 必须是 2 的幂（新，最容易踩）。**
地址映射器 `OneLevelInterleave` 用 `calc_log2(channel)` 定通道位宽。`channel=5` 的结果和 `channel=4` **逐位相同**，`channel=10` 等同 `8`，`20/24` 等同 `16`。写非 2 的幂不会报错，只会给你重复数据，曲线看着“点很多”，其实是假的。`dramcfg.make_cfg` 已加断言拦住。

**⑦ 每个点必须独立进程（新）。**
booksim 的 `rand()` 只在 `noc_main()` 里 `srand(1234)`；走库调用路径不播种。同进程连跑第二点会消耗共享 RNG 状态，产生 **0.4% 假波动**（实测 `9642` → `9683` cycles）。这足以让你的“边际收益 < 3%”阈值误判。`sim.py` 用 `subprocess` 每点一个新进程，天然满足；**不要图快改成同进程循环**。

---

## 11. 关于“单参数”这件事，汇报时必须说清的三点

1. **只有 7 个真实候选点**，因为 `channel` 必须是 2 的幂。给 20 个点必然掺假。
2. **`rate` 不是带宽轴**：固定周期数只改 `rate` 是在缩短 DRAM 事务的绝对时间；按物理正确方式同步放大周期数后，1000/2000 Mbps 与 500 Mbps 的延迟**完全一致**（`3.34e-4`）。别把它当第二个带宽点。
3. **代价函数必须带面积项**，否则 `cost ≈ (L/L0)²` 单调递减，最优永远是最右边那个点，Agent 无事可做。加了 `A/A0 = channel/8`（baseline ch=8）才有内部极小值（实测最低在 ch=16 / 1024 GB/s = 0.9216）。

在 1D 单峰曲线上，LLM 赢不了二分/黄金分割。它唯一能赢的是线性外推这类笨基线；真正的价值在 N 维。**这句话要写进汇报，不要回避。**

---

## 12. 与 Beacon 的关系（一句话）

**这份实现不 import `chiplet_tuner` 的任何模块。** 它只借用了三个思想：指纹去重、合法性门控、best-so-far。Beacon 的 `search_space.py` / `analysis_tools.py` / `solution.py` 一行都不碰。
