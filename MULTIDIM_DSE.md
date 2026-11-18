# AgentDSE 多维度扩展手册

> 这份文档把现有的 **1 维 DSE（只搜 DRAM 带宽）** 扩展成 **N 维 DSE（先做 bw + mac_num + vec_num 三维）**。
> 写法沿用 `DEVELOPMENT_MANUAL.md`：每一步都说清楚 **改哪个文件 / 改哪里 / 怎么验证这一步真的成了**。
>
> **不改 ATLAS 源码。** 所有改动都在 `agentdse/` 自己的代码里。
> **不重写架构。** `store.py` / `tools.py` / `agent.py` / `sim.py` / `prompts.py` 的分层和命名原样保留，只换接口。

---

## 0. 先说结论：为什么这次改动比想象中少

现有代码里真正绑死"一维"的地方只有 **5 处**，其余全是"顺手这么写"，不是架构需要：

| 位置 | 现状 | 性质 |
|---|---|---|
| `store.py` | `fp = f"bw{record['bw_gbps']}"` | 主键生成规则 |
| `store.py` | `has_bw(bw)` | 命名 + 查询条件 |
| `agent.py` | `propose()` 返回 `{"bw_gbps": int}` | 返回值类型 |
| `tools.py` | `tool_trend` 按 bw 排序 | 单轴语义 |
| `prompts.py` | "One tunable parameter: DRAM bandwidth" | 文案 |

所以本手册的策略是：

1. **新增** 一个 `agent_dse/dims.py`，把所有"维度"相关的知识抽出来，**不写死任何维度名**。
2. 其他文件只改"调 dims.py"和"收 dict 而不是 int"。
3. 以后加第 4 维，**只改 `config/space.json`（加一段配置）+ 可能加一行系数**，代码基本不动。

---

## 0.5 三个必须先知道的实测事实（都有代码证据）

这一节是第 3 条要求"验证 ATLAS 自己会不会做 DSE"的答复。**三条全部属实或需要修正**，逐条给证据。

### 事实 A：ATLAS 确实会自己搜索最优 tiling —— 属实

`ATLAS-MICRO-2026-main/frontend/auto/tiling_explorer_cloud.py:297-313`，`explore_gemm_comp()` 枚举所有合法 `(tM, tK, tN)`：

```python
tM_list = get_factors(operator.M, min_tM)      # min_tM = 8
tK_list = get_factors(operator.K, min_tK)      # min_tK = 512
tN_list = get_factors(operator.N, min_tN)      # min_tN = 8
...
tiling_factor_combination_list = list(product(tM_list, tK_list, tN_list))
```

`get_factors` 在 `frontend/util.py:98`，就是枚举因子。

然后按每核 SRAM 容量剪枝（`tiling_explorer_cloud.py:316-326`）：

```python
buffer_size = chip_config.buffer_config.buffer_size * _KB      # _KB = 1024, util.py:10
if element_size * (tM*tK + tK*tN + 2*tM*tN) > buffer_size / 2:  # Double buffer
    continue
```

**每个幸存组合都真跑一次 C++ 仿真**（`tiling_explorer_cloud.py:127-236`，worker 内层循环）：

```python
for tiling_factor_combination in tiling_factor_combination_list:
    ...
    chip = Chip(chip_config_path, comp_description_path, data_placement_config_path)
    performance = chip.simulate()
    del chip
    latency = performance.e2e_stats.e2e_cycles
    if latency < opt_latency:
        opt_latency = latency
        opt_comp_description = (comp_description, (tM, tK, tN))
```

最后取全局最小（`tiling_explorer_cloud.py:364-366`）：

```python
for result in results_list:
    if result["latency"] < opt_latency:
        opt_latency = result["latency"]
        opt_comp_description = result
```

**结论**：你只给硬件配置，tiling 由 ATLAS 搜到最优。**变量单一性由仿真器保证，你不需要在 DSE 层再管 tiling。** 这也解释了一个点为什么要跑 30 秒——里面其实跑了几百次 C++ `simulate()`。

规模（llama3_8b / tp=8 / core_num=16 下实测因子数）：单个 GEMM 大约 **196~245** 个组合，剪枝后再跑。

### 事实 B：tiling 缓存键不含硬件 —— **这是加维度后最大的坑**

`tiling_explorer_cloud.py:279-291`：

```python
cache_file_path = os.path.join(cur_op_dir, "gemm_opt_general_layout.yaml")
if os.path.exists(cache_file_path):
    with open(cache_file_path, "r") as f:
        cached = yaml.safe_load(f)
    if (cached is not None
        and cached.get("M") == operator.M
        and cached.get("K") == operator.K
        and cached.get("N") == operator.N
        and cached.get("B") == operator.B
        and cached.get("element_size") == element_size):
        print(f"[Cache hit] {operator.name}: tiling factors {cached['tiling_factors']}")
        return {                                  # ← 直接返回旧结果，连 latency 都是旧的
            "latency": cached["latency"],
            "comp_description_list": cached["comp_description_list"],
            "tiling_factors": tuple(cached["tiling_factors"]),
        }
```

缓存键只有 **`M / K / N / B / element_size`**，这五个全由**模型和并行策略**决定，与 `mac_num / vec_num / buffer_size / frequency` 无关。

而 `mac_num` 恰恰**会影响**最优 tiling，因为延迟是这么算的（`simulator/src/core/matrix_unit.cpp:64`）：

```cpp
return uint64_t(std::ceil((double)cur_mac_count / mac_num));
```

`vector_unit.cpp:64` 同理用 `vec_num`。

**所以会发生这个静默错误**：

```
第一次：mac_num=8192  → 搜出最优 tiling，写缓存，latency=572
第二次：mac_num=4096  → M/K/N/B 没变 → 命中缓存 → 返回 tiling 和 latency=572
        但 mac 减半了，真实 latency 应该翻倍
```

结果就是：**你的 `mac_num` 维度在数据里显示"完全没影响"**，而且不报错、不警告，日志里只多一行 `[Cache hit]`。

**为什么现在不会踩**：现在每个点写在独立目录

```
data/artifacts/bw1024/atlas_artifacts/operator_tling_exploration/qkv_proj/gemm_opt_general_layout.yaml
data/artifacts/bw2048/atlas_artifacts/operator_tling_exploration/qkv_proj/gemm_opt_general_layout.yaml
```

`tag=f"bw{bw}"` 一变目录就变，缓存天然隔离。

**本手册的对策：让 `tag` 包含所有维度。** 一行 `tag = fingerprint(point)` 就解决，ATLAS 源码一个字都不用改。

代价：每个点都要重搜 tiling，仍然是约 30 秒/点，不能靠缓存加速。**这是正确的取舍**——共享缓存省时间，但会静默污染结果；隔离目录慢一点，但每个数字都是真的。

> ⚠️ 绝对不要为了复用缓存去共享 `tag`。一旦共享，就必须改 ATLAS 的缓存键（加 `mac_num` 等），那就是动源码了。

### 事实 C：`area` 不参与仿真，`power` 参与 —— cost 里的面积只能我们自己算

**`area` 只被打印，不参与任何计算**（`simulator/src/common/arch.h:167` 是唯一用处）：

```cpp
os << std::setw(indent) << "        Area: " << matrix_config.area << std::endl;
```

全仓库 `grep -rn "\.area" simulator/src/` 只有 `arch.h` 的 `display_stats()` 命中。

**`power` 才是真正参与能量计算的**（`simulator/src/core/core_array.cpp:14-19`）：

```cpp
stats.controller_energy = core_num * elapsed_seconds * arch_config->controller_config.power;
stats.matrix_energy     = core_num * elapsed_seconds * arch_config->matrix_config.power;
stats.vector_energy     = core_num * elapsed_seconds * arch_config->vector_config.power;
stats.buffer_energy     = core_num * elapsed_seconds * arch_config->buffer_config.power;
stats.dram_energy       = core_num * elapsed_seconds * arch_config->dram_config.power;
stats.noc_energy        = core_num * elapsed_seconds * (arch_config->has_noc ? arch_config->noc_config.power : 0.0);
```

**这意味着**：只改 `mac_num` 而不改 `power`，能量几乎不变（只有延迟变），能量维度就失去了 trade-off 的意义。所以 `sim.py` 必须让 `power` 跟着 `mac_num`/`vec_num` 缩放。

### 面积/功耗系数从哪来

ATLAS 自己在 `ATLAS-MICRO-2026-main/configs/architecture/chip/cloud/matrix_vector/` 下给了 5 组 preset。我把它们全部算了一遍：

| preset | mac_num | matrix area | matrix power | vec_num | vector area | vector power |
|---|---|---|---|---|---|---|
| matrix4_vector1 | 5440 | 9.2972 | 2.2135 | 2720 | 7.0834 | 2.1338 |
| matrix8_vector1 | 6528 | 11.1566 | 2.6562 | 1632 | 4.2500 | 1.2803 |
| matrix16_vector1 | 7168 | 12.2504 | 2.9167 | 896 | 2.3333 | 0.7029 |
| matrix32_vector1 | 7680 | 13.1255 | 3.1250 | 480 | 1.2500 | 0.3766 |
| matrix64_vector1 | 7680 | 13.1255 | 3.1250 | 240 | 0.6250 | 0.1883 |

`area/mac_num`、`area/vec_num`、`power/mac_num`、`power/vec_num` 在 5 组里**小数点后 5 位完全一致**：

```
MAC_AREA_PER_UNIT  = 1.709044e-3   mm² / MAC
MAC_POWER_PER_UNIT = 0.406899e-3   W   / MAC
VEC_AREA_PER_UNIT  = 2.604164e-3   mm² / lane
VEC_POWER_PER_UNIT = 0.784527e-3   W   / lane
```

固定项（同样 5 组全一致）：

```
CONTROLLER_AREA = 3.665     mm²  / core
CONTROLLER_POWER= 0.725     W    / core
BUFFER_AREA     = 15.7843   mm²  / core
BUFFER_POWER    = 5.0869    W    / core
NOC_AREA        = 6.597     mm²  / chip
NOC_POWER       = 0.4754    W    / chip
DRAM_AREA_PER_CHANNEL = 4.998/16 = 0.312375  mm² / channel
DRAM_POWER      = 5.333     W    / chip
```

> **注意**：上一轮我说过"core 总面积固定，mac 和 vec 互相挤占"，**这句是错的**，在此更正。
> ATLAS 的 5 个 preset 是人为设计的"算力配比方案"，**代码里没有任何机制约束总面积守恒**。
> 实测 `m+v` 之和在 13.7505 ~ 16.3806 之间浮动，面积就是简单可加的。
> 所以你可以同时调大 mac 和 vec，面积线性增长——这正好是 DSE 需要的惩罚项。

用这套系数可以精确复现 ATLAS 的 preset，说明口径一致：

```
mac_num=6528 → 6528 × 1.709044e-3 = 11.1566   ✓ 与 matrix8_vector1.yaml 一致
vec_num=1632 → 1632 × 2.604164e-3 = 4.2500    ✓ 一致
```

整芯片面积公式（本手册采用）：

```
core_area    = CTRL_A + BUF_A + mac_num*MAC_A + vec_num*VEC_A
chip_area(p) = core_num * core_area + channel * DRAM_A_PER_CH + NOC_A
```

验算（core_num=16）：

| 点 | 面积 |
|---|---|
| (ch=16, mac=6144, vec=512) ← **baseline** | 512.1230 mm² |
| (ch=8, mac=6144, vec=512) | 509.6240 mm² |
| (ch=2, mac=4096, vec=128) | 435.7478 mm² |
| (ch=32, mac=8192, vec=1024) | 594.4562 mm² |

---

## 1. 设计：把"维度"抽象成一个配置

### 核心概念

**一个点（point）就是一个 `dict[str, int]`**：

```python
{"bw_gbps": 1024, "channel": 16, "density_gb": 4, "mac_num": 6144, "vec_num": 768}
```

- `bw_gbps / mac_num / vec_num` 是**被搜索的维度**（`dims`）。
- `channel / density_gb` 是 `bw_gbps` 的**随行参数**（bw 的物理实现），不单独搜索，但必须塞进 point 传给仿真器。

**point 的构造方式**：每个维度的每一档是一个小 dict，把各维度选中的档 merge 起来就是 point。

```python
# bw_gbps 的某一档
{"bw_gbps": 1024, "channel": 16, "density_gb": 4}
# mac_num 的某一档
{"mac_num": 6144}
# vec_num 的某一档
{"vec_num": 768}
# merge 之后
{"bw_gbps": 1024, "channel": 16, "density_gb": 4, "mac_num": 6144, "vec_num": 768}
```

这个设计的好处：**加第 4 维时，只要它的档本身能用一个 dict 描述，代码完全不用动。**

### fingerprint

```python
def fingerprint(point, names):
    return "|".join(f"{n}={point[n]}" for n in names)
# → "bw_gbps=1024|mac_num=6144|vec_num=768"
```

- 用维度名而不是缩写，自解释，不会撞车。
- 顺序由 `space.json` 里 `dims` 的顺序决定，稳定。
- 这个字符串同时用作 **sqlite 主键** 和 **artifact 目录名**（见事实 B）。

> 目录名里会出现 `|` 和 `=`，Linux 下合法，但不好看也不方便 shell 补全。
> 所以建议目录用 `tag`，主键用 `fingerprint`：
> ```python
> tag = "bw1024_mac6144_vec768"        # 给目录用
> fp  = "bw_gbps=1024|mac_num=6144|vec_num=768"   # 给数据库用
> ```
> 两者都从同一个 point 生成，见 Step 2 的 `dims.py`。

### 新增可用维度

候选值我按"低/中/高"三个档取（3×3×3 是第一节演示，实际配 5×4×6=120 点）：

| 维度 | 候选值 | 说明 |
|---|---|---|
| `bw_gbps` | 256 / 512 / 1024 / 2048 / 4096 | 对应 channel 4/8/16/32/64 |
| `mac_num` | 2048 / 4096 / 6144 / 8192 | MAC 阵列规模 |
| `vec_num` | 64 / 128 / 256 / 512 / 768 / 1024 | 向量通道数 |

> 都是 2 的幂或 2、3 的倍数，**避免奇怪的因子组合导致 tiling 搜索变慢**。
> `channel` 必须保持 2 的幂（原手册结论 A：`OneLevelInterleave` 用 `calc_log2(channel)`，
> 非 2 的幂会产生逐位相同的假点）。

---

## 2. 第一步：改 `config/space.json`

**这是唯一一个"加维度只改这里"的文件。** 完整替换成：

```json
{
    "objective": "cost = (latency/base_latency) * (energy/base_energy) * (area/base_area)",

    "dims": [
        {
            "name": "bw_gbps",
            "levels": [
                {"bw_gbps": 256,  "channel": 4,  "density_gb": 16},
                {"bw_gbps": 512,  "channel": 8,  "density_gb": 8},
                {"bw_gbps": 1024, "channel": 16, "density_gb": 4},
                {"bw_gbps": 2048, "channel": 32, "density_gb": 2},
                {"bw_gbps": 4096, "channel": 64, "density_gb": 1}
            ]
        },
        {
            "name": "mac_num",
            "levels": [
                {"mac_num": 2048},
                {"mac_num": 4096},
                {"mac_num": 6144},
                {"mac_num": 8192}
            ]
        },
        {
            "name": "vec_num",
            "levels": [
                {"vec_num": 64},
                {"vec_num": 128},
                {"vec_num": 256},
                {"vec_num": 512},
                {"vec_num": 768},
                {"vec_num": 1024}
            ]
        }
    ],

    "seed_n": 27,
    "seed_strategy": "farthest",

    "core_num": 16,
    "fixed_hardware": {
        "buffer_size": 3072,
        "buffer_read_bw": 8192,
        "buffer_write_bw": 8192,
        "frequency": 1000
    }
}
```

**字段说明**

| 字段 | 含义 |
|---|---|
| `dims[].name` | 维度名，必须出现在 `levels` 的每个 dict 里 |
| `dims[].levels` | 该维度的候选档，**每档是一个 dict**（可携带随行参数） |
| `seed_n` | 开跑前先仿真多少个点作为 Agent 的起点 |
| `seed_strategy` | `"farthest"` = 从 baseline 出发按归一化距离由远及近铺开；`"random"` = 固定种子随机 |
| `core_num` | 从 `run.json` 挪到这里，因为它和面积计算绑定 |
| `fixed_hardware` | 不参与搜索、但每个点都要传给仿真的字段 |

**搜索空间规模**：5 × 4 × 6 = **120 个点**。

> 全枚举约 120 × 35s ≈ 70 分钟。所以**不要全跑**，靠 `seed_n` + Agent 迭代。

**baseline 的定义**：每个维度取 `levels[len(levels)//2]`。

```
bw_gbps: 5 档 → index 2 → 1024
mac_num: 4 档 → index 2 → 6144
vec_num: 6 档 → index 3 → 512
→ baseline = (1024, 6144, 512)
```

它的 cost 恒为 **1.0**，其余点相对它比较。`cost < 1` 就是比基线好。

### 同时要改 `config/run.json`

把 `core_num` 和 `fixed_hardware` 里的 `mac_num`/`vec_num` 删掉（已移到 `space.json`），只留下模型和仿真参数：

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

**验证**：`python -c "import json;print(json.load(open('config/space.json'))['dims'][0]['levels'][2])"` 应输出 `{'bw_gbps': 1024, 'channel': 16, 'density_gb': 4}`。

---

## 3. 第二步：新增 `agent_dse/dims.py`（扩展接口的核心）

**这是整个改造的关键文件。它不包含任何 `bw` / `mac` / `vec` 字面量**，全部从配置读。

创建 `agent_dse/dims.py`：

```python
from __future__ import annotations

from itertools import product
from typing import Any, Dict, List, Optional


class Space:
    """维度无关的搜索空间。

    只认 config 里的结构，不知道任何具体维度名。
    加维度时这个文件不用改。
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.dims: List[Dict[str, Any]] = cfg["dims"]
        self.name: List[str] = [d["name"] for d in cfg["dims"]]
        self.names = self.name          # 别名：部分调用点用 names
        self.core_num: int = int(cfg.get("core_num", 16))
        self.fixed_hardware: Dict[str, Any] = dict(cfg.get("fixed_hardware", {}))

    # ---------- 基础查询 ----------
    def levels(self, name: str) -> List[Dict[str, Any]]:
        for d in self.dims:
            if d["name"] == name:
                return d["levels"]
        raise KeyError(f"unknown dimension: {name}")

    def values(self, name: str) -> List[int]:
        return [lvl[name] for lvl in self.levels(name)]

    def all_points(self) -> List[Dict[str, Any]]:
        """笛卡尔积，每个点 = 各维度选中档的 merge。"""
        out: List[Dict[str, Any]] = []
        for combo in product(*[d["levels"] for d in self.dims]):
            point: Dict[str, Any] = {}
            for lvl in combo:
                point.update(lvl)
            point.update(self.fixed_hardware)
            out.append(point)
        return out

    # ---------- 标识 ----------
    def fingerprint(self, point: Dict[str, Any]) -> str:
        """数据库主键。稳定、自解释、加维度后旧格式自然失效。"""
        return "|".join(f"{n}={point[n]}" for n in self.names)

    def tag(self, point: Dict[str, Any]) -> str:
        """artifact 目录名。同时保证 tiling 缓存按点隔离（见事实 B）。"""
        return "_".join(f"{n}{point[n]}" for n in self.names)

    # ---------- 基线与合法性 ----------
    def baseline(self) -> Dict[str, Any]:
        """每维取中间档。"""
        point: Dict[str, Any] = {}
        for d in self.dims:
            lvl = d["levels"][len(d["levels"]) // 2]
            point.update(lvl)
        point.update(self.fixed_hardware)
        return point

    def is_legal(self, point: Dict[str, Any]) -> bool:
        if not isinstance(point, dict):
            return False
        for name in self.names:
            if name not in point:
                return False
            if point[name] not in self.values(name):
                return False
        return True

    # ---------- 给 prompt 用的文本 ----------
    def describe(self) -> str:
        lines = [f"There are {len(self.names)} tunable dimensions "
                 f"({len(self.all_points())} legal combinations in total):"]
        for d in self.dims:
            vals = self.values(d["name"])
            lines.append(f"  - {d['name']}: {vals}")
        lines.append("")
        lines.append("Extra fixed fields carried by every point "
                     "(do NOT propose these, they are not tunable):")
        lines.append(f"  {self.fixed_hardware}")
        return "\n".join(lines)

    def point_brief(self, point: Dict[str, Any]) -> str:
        return " ".join(f"{n}={point[n]}" for n in self.names)
```

### 逐段解释

| 方法 | 为什么这么写 |
|---|---|
| `all_points()` | 用 `product` + `dict.update` **merge 各档**。所以任何维度只要能用 dict 描述，就自动支持。 |
| `fingerprint()` / `tag()` | 两个都从 `self.names` 派生，**加维度时自动带上**，不需要改代码。 |
| `baseline()` | `len//2` 对 3 档给 index 1、5 档给 index 2、4 档给 index 2。通用。 |
| `is_legal()` | 只检查"每个维度名都在且值合法"。**不检查维度个数**，所以加维度后自动生效。 |
| `describe()` | 直接喂给 prompt，LLM 看到的就是"有 N 个维度、各自候选值"，换维度后文案自动变。 |

### 验证

```bash
python -c "
import json
from agent_dse.dims import Space
sp = Space(json.load(open('config/space.json')))
print('dims    :', sp.names)
print('total   :', len(sp.all_points()))
print('baseline:', sp.point_brief(sp.baseline()))
p = sp.all_points()[0]
print('fp      :', sp.fingerprint(p))
print('tag     :', sp.tag(p))
"
```

期望输出：

```
dims    : ['bw_gbps', 'mac_num', 'vec_num']
total   : 120
baseline: bw_gbps=1024 mac_num=6144 vec_num=512
fp      : bw_gbps=256|mac_num=2048|vec_num=64
tag     : bw_gbps256_mac_num2048_vec_num64
```

---

## 4. 第三步：新增 `agent_dse/costfn.py`（面积/功耗模型）

把 cost 从"测试脚本里的私有函数"提到正式模块，因为现在它要处理多维。

创建 `agent_dse/costfn.py`：

```python
from __future__ import annotations

from typing import Any, Dict

# 系数全部从 ATLAS 自己的 preset 反推，见手册 0.5 节
MAC_AREA_PER_UNIT  = 1.709044e-3   # mm^2 / MAC
MAC_POWER_PER_UNIT = 0.406899e-3   # W / MAC
VEC_AREA_PER_UNIT  = 2.604164e-3   # mm^2 / vector lane
VEC_POWER_PER_UNIT = 0.784527e-3   # W / lane

CONTROLLER_AREA  = 3.665
CONTROLLER_POWER = 0.725
BUFFER_AREA      = 15.7843
BUFFER_POWER     = 5.0869
NOC_AREA         = 6.597
NOC_POWER        = 0.4754
DRAM_AREA_PER_CHANNEL = 4.998 / 16
DRAM_POWER       = 5.333

# 每个维度"每单位"贡献多少面积/功耗。加新维度时在这里加一行即可；
# 不贡献面积/功耗的维度（比如某种策略开关）完全不用登记。
AREA_PER_UNIT: Dict[str, float] = {
    "mac_num": MAC_AREA_PER_UNIT,
    "vec_num": VEC_AREA_PER_UNIT,
}
POWER_PER_UNIT: Dict[str, float] = {
    "mac_num": MAC_POWER_PER_UNIT,
    "vec_num": VEC_POWER_PER_UNIT,
}


def chip_area(point: Dict[str, Any], core_num: int) -> float:
    """整芯片面积 (mm^2)。

    core 内部：controller + buffer + sum(各维度按单位面积累加)
    芯片级：  core_num * 上述 + DRAM(按 channel) + NoC
    """
    per_core = CONTROLLER_AREA + BUFFER_AREA
    for name, coeff in AREA_PER_UNIT.items():
        if name in point:
            per_core += float(point[name]) * coeff
    total = core_num * per_core
    total += float(point.get("channel", 0)) * DRAM_AREA_PER_CHANNEL
    total += NOC_AREA
    return total


def unit_power(point: Dict[str, Any], name: str) -> float:
    """某个可缩放部件的功耗 (W)。矩阵/向量单元按规模线性缩放。"""
    coeff = POWER_PER_UNIT.get(name)
    if coeff is None:
        return 0.0
    return float(point.get(name, 0)) * coeff


def cost(point: Dict[str, Any],
         metrics: Dict[str, Any],
         base_point: Dict[str, Any],
         base_metrics: Dict[str, Any],
         core_num: int) -> float:
    """(L/L0) * (E/E0) * (A/A0)，越低越好。baseline 的 cost 恒为 1.0。"""
    l_ratio = metrics["latency_s"] / base_metrics["latency_s"]
    e_ratio = metrics["energy_j"]  / base_metrics["energy_j"]
    a_ratio = chip_area(point, core_num) / chip_area(base_point, core_num)
    return l_ratio * e_ratio * a_ratio
```

### 为什么 `AREA_PER_UNIT` 用 `dict` 而不是 `if/elif`

这就是你要的扩展接口：**加第 4 维时，如果它消耗面积，在这里加一行；如果不消耗，一行都不用加。**

```python
# 未来加一个 SRAM 大小维度
AREA_PER_UNIT["sram_kb"] = 0.0038
POWER_PER_UNIT["sram_kb"] = 0.0011
```

### 验证

```bash
python -c "
import json
from agent_dse.dims import Space
from agent_dse import costfn as cf
sp = Space(json.load(open('config/space.json')))
b = sp.baseline()
print('baseline area =', round(cf.chip_area(b, sp.core_num), 4), 'mm^2')
print('baseline mat P=', round(cf.unit_power(b,'mac_num'),4), 'W')
print('baseline vec P=', round(cf.unit_power(b,'vec_num'),4), 'W')
"
```

期望：

```
baseline area = 522.7896 mm^2
baseline mat P= 2.5 W
baseline vec P= 0.6025 W
```

---

## 5. 第四步：改 `agent_dse/sim.py`

改动 **3 处**，其余不动。

### 5.1 `build_system_yaml` 从 point 取硬件，并缩放 power

把签名里的 `fixed_hardware: Dict` 换成 `point: Dict`，并加 power 计算。找这一行：

```python
arch["core"]["matrix"]["mac_num"] = int(fixed_hardware["mac_num"])
```

整段（从 `arch["frequency"]=...` 到 `arch["core"]["buffer"]["write_bw"]=...`）替换成：

```python
    arch["frequency"] = float(point["frequency"])
    arch["core"]["matrix"]["mac_num"] = int(point["mac_num"])
    arch["core"]["vector"]["vec_num"]  = int(point["vec_num"])
    arch["core"]["buffer"]["buffer_size"] = int(point["buffer_size"])
    arch["core"]["buffer"]["read_bw"]     = float(point["buffer_read_bw"])
    arch["core"]["buffer"]["write_bw"]    = float(point["buffer_write_bw"])

    # 关键：power 必须随规模缩放，否则能量维度是平的（见 0.5 节事实 C）
    arch["core"]["matrix"]["power"] = costfn.unit_power(point, "mac_num")
    arch["core"]["vector"]["power"] = costfn.unit_power(point, "vec_num")
```

同时在文件顶部 import 里加：

```python
from . import costfn
```

> **如果你嫌 power 太小**（baseline 下 matrix 只有 2.5 W，而 buffer 是 5.0869 W），
> 说明 ATLAS preset 里的 `power` 就是按"每 MAC 0.4 mW"这个量级给的工作负载功率。
> **保持一致，不要自己放大**，否则和 ATLAS 论文数据不可比。

### 5.2 `run_once` 改成收 point

把整个 `run_once` 签名和开头替换成：

```python
def run_once(
    point: Dict[str, Any],
    run_cfg: Dict[str, Any],
    space: "Space",
    tag: Optional[str] = None,
) -> Dict[str, Any]:
    """跑一个点。tag 必须唯一标识该点，否则 tiling 缓存会串味。"""
    tag = tag or space.tag(point)

    dram_yaml = dramcfg.write_cfg(
        channel=int(point["channel"]),
        density_gb=int(point["density_gb"]),
    )
```

后面的 `build_system_yaml` 调用改成：

```python
    system_yaml = build_system_yaml(
        out_dir,
        dram_yaml=dram_yaml,
        point=point,
        core_num=int(space.core_num),
    )
```

结尾的 payload 补字段，把**所有维度值**带进记录（这样 `store` 不需要为每个维度建列）：

```python
    payload = json.loads(result_path.read_text())
    # 通用：point 的所有字段（维度值 + 随行参数 + fixed_hardware）都进记录。
    # 加第 4 维时这里不用改。用 setdefault 是为了不让维度名意外覆盖仿真指标。
    for k, v in point.items():
        payload.setdefault(k, v)
    payload["fingerprint"] = space.fingerprint(point)
    payload["run_dir"] = str(out_dir)
    return payload
```

> **为什么用 `setdefault` 而不是 `payload.update(point)`**：如果将来某个维度恰好叫
> `latency_s` 之类的名字，`update` 会把仿真结果覆盖掉，`setdefault` 不会。
> 这条在加第 4 维时很重要——你不需要回来检查有没有撞名。

顶部加类型导入（避免循环依赖，用字符串注解）：

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .dims import Space
```

### 5.3 为什么这样就能解决缓存问题

`space.tag(point)` 对三维会生成 `bw_gbps1024_mac_num6144_vec_num512`。`ARTIFACT_DIR / tag` 一变，`atlas_artifacts/` 下的 `gemm_opt_general_layout.yaml` 就是新的空目录，**必然重搜 tiling**。

`ATLAS-MICRO-2026-main` 一个字节都不用改。

### 验证

```bash
python -c "
import json
from agent_dse.dims import Space
from agent_dse.sim import run_once
sp = Space(json.load(open('config/space.json')))
run_cfg = json.load(open('config/run.json'))
p = sp.baseline()
rec = run_once(p, run_cfg, sp, tag='smoke_baseline')
print({k: rec[k] for k in ('bw_gbps','latency_s','energy_j','channel')})
print('fingerprint =', rec['fingerprint'])
"
```

期望：约 30~40 秒后打印出 `latency_s ≈ 3.42e-4`、`energy_j ≈ 0.0444`，`fingerprint` 形如 `bw_gbps=1024|mac_num=6144|vec_num=512`。

> 注意：这次 `energy_j` 会和旧版**不一样**，因为 matrix/vector 的 power 从写死的 1.1/1.2 变成了 2.5/0.6025。这是正确行为。

---

## 6. 第五步：改 `agent_dse/store.py`

### 6.1 新 schema：不再为每个维度建列

把 `SCHEMA` 整个替换成：

```python
SCHEMA = """
CREATE TABLE IF NOT EXISTS trials (
    fingerprint TEXT PRIMARY KEY,
    status      TEXT NOT NULL,
    cost        REAL,
    latency_s   REAL,
    energy_j    REAL,
    record      TEXT NOT NULL,
    created_at  TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS decisions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    round        INTEGER NOT NULL,
    chosen_fp    TEXT NOT NULL,
    chosen_pt    TEXT NOT NULL,
    reason       TEXT,
    evidence     TEXT,
    created_at   TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS best (
    id           INTEGER PRIMARY KEY CHECK (id = 1),
    fingerprint  TEXT NOT NULL,
    point        TEXT NOT NULL,
    cost         REAL NOT NULL,
    updated_at   TEXT DEFAULT CURRENT_TIMESTAMP
);
"""
```

**为什么删掉 `bw_gbps/channel/density_gb` 三列**：加第 4 维时你不想再改表结构。所有维度值都在 `record` 的 JSON 里，排序/筛选在 Python 里做（点数只有几十个，性能完全够）。

### 6.2 `put_trial` 用调用方给的 fingerprint

```python
    def put_trial(self, record: Dict[str, Any]) -> bool:
        """插入。返回 True 表示新点，False 表示已存在。"""
        fp = record["fingerprint"]
        try:
            self.conn.execute(
                "INSERT INTO trials (fingerprint,status,cost,latency_s,energy_j,record)"
                " VALUES (?,?,?,?,?,?)",
                (
                    fp,
                    record.get("status", "ok"),
                    record.get("cost"),
                    record.get("latency_s"),
                    record.get("energy_j"),
                    json.dumps(record),
                ),
            )
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
```

### 6.3 `has_bw` → `has_point`

```python
    def has_point(self, fingerprint: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM trials WHERE fingerprint=?", (fingerprint,)
        ).fetchone()
        return row is not None
```

### 6.4 `all_trials` 不再按 bw 排序

```python
    def all_trials(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT record FROM trials ORDER BY created_at"
        ).fetchall()
        return [json.loads(r["record"]) for r in rows]

    def ok_trials(self) -> List[Dict[str, Any]]:
        return [r for r in self.all_trials() if r.get("status") == "ok"]

    def get_trial(self, fingerprint: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT record FROM trials WHERE fingerprint=?", (fingerprint,)
        ).fetchone()
        return json.loads(row["record"]) if row else None
```

### 6.5 `put_decision` / `update_best` / `best`

```python
    def put_decision(self, rnd: int, point: Dict[str, Any],
                     fingerprint: str, reason: str, evidence: str) -> None:
        self.conn.execute(
            "INSERT INTO decisions (round,chosen_fp,chosen_pt,reason,evidence)"
            " VALUES (?,?,?,?,?)",
            (rnd, fingerprint, json.dumps(point), reason, evidence),
        )
        self.conn.commit()

    def update_best(self, point: Dict[str, Any],
                    fingerprint: str, cost: float) -> str:
        row = self.conn.execute("SELECT cost FROM best WHERE id=1").fetchone()
        if row is None:
            self.conn.execute(
                "INSERT INTO best (id,fingerprint,point,cost) VALUES (1,?,?,?)",
                (fingerprint, json.dumps(point), cost),
            )
            self.conn.commit()
            return "inserted"
        if cost < row["cost"]:
            self.conn.execute(
                "UPDATE best SET fingerprint=?, point=?, cost=? WHERE id=1",
                (fingerprint, json.dumps(point), cost),
            )
            self.conn.commit()
            return "updated"
        return "kept"

    def best(self) -> Optional[Dict[str, Any]]:
        row = self.conn.execute("SELECT * FROM best WHERE id=1").fetchone()
        if not row:
            return None
        return {
            "fingerprint": row["fingerprint"],
            "point": json.loads(row["point"]),
            "cost": row["cost"],
        }
```

> **注意语义变化**：`best()` 现在返回 `{"fingerprint", "point", "cost"}`，不再是 `{"bw_gbps", "cost"}`。
> 所有调用点（`agent.py`、`tools.py`、测试脚本）都要跟着改。

### 6.6 旧库要删

新 schema 和旧库不兼容。跑之前先：

```bash
rm -f data/trials.db data/agent_test.db data/test.db
```

**验证**：

```bash
python -c "
import json
from agent_dse.store import Store
from agent_dse.dims import Space
sp = Space(json.load(open('config/space.json')))
s = Store()
p = sp.baseline()
fp = sp.fingerprint(p)
print('insert :', s.put_trial({'fingerprint': fp, 'status':'ok', 'cost':1.0,
                               'latency_s':3.3e-4, 'energy_j':0.04, **p}))
print('dup    :', s.put_trial({'fingerprint': fp, 'status':'ok', 'cost':1.0, **p}))
print('has    :', s.has_point(fp))
s.update_best(p, fp, 1.0)
print('best   :', s.best())
"
```

期望：`True / False / True`，最后 `best` 里 `cost: 1.0`。

---

## 7. 第六步：改 `agent_dse/tools.py`

四个工具都要从"bw"泛化成"point"。**函数名保持不变**，只改参数和实现。

### 7.1 `tool_lookup` —— 几乎不用改

只把 `_compact` 扩展成把维度值都带上：

```python
def _compact(r: Dict[str, Any], names: List[str]) -> Dict[str, Any]:
    out = {n: r.get(n) for n in names}
    out.update({
        "cost": r.get("cost"),
        "latency_s": r.get("latency_s"),
        "energy_j": r.get("energy_j"),
        "dram_util": r.get("dram_util"),
        "matrix_util": r.get("matrix_util"),
        "dram_stall_cycles": r.get("dram_stall_cycles"),
    })
    return out
```

`tool_lookup` 里对 `_compact` 的调用改成 `_compact(r, names)`，并在签名加 `names`。

### 7.2 `tool_inspect` —— 从 bw 改成 fingerprint

```python
def tool_inspect(store: Store, fingerprint: str) -> Dict[str, Any]:
    rec = store.get_trial(fingerprint)
    if rec is None:
        return {"tool": "inspect_point", "found": False,
                "fingerprint": fingerprint,
                "note": "this point has not been simulated yet"}
    return {"tool": "inspect_point", "found": True, "point": rec}
```

### 7.3 `tool_trend` —— 从单轴差分改成逐轴差分

这是唯一有"语义"改动的工具。**一维时"相邻点"是天然定义的，多维时不存在全局排序**，所以要按维度分别算，并且只在"其他维度取值相同"的点之间比。

```python
def tool_trend(store: Store, space) -> Dict[str, Any]:
    """对每个维度，找出"除该维外其余维度都相同"的点组，按该维排序后算一阶差分。

    三维空间里这种"同 profile 组"常常只有 2 个点甚至 1 个点，
    所以额外给一个 global 段：按 cost 排序的前若干点，供 LLM 直接比较。
    """
    rows = store.ok_trials()
    names = space.names
    per_axis: Dict[str, List[Dict[str, Any]]] = {}

    for target in names:
        others = [n for n in names if n != target]
        groups: Dict[tuple, List[Dict[str, Any]]] = {}
        for r in rows:
            key = tuple(r[n] for n in others)
            groups.setdefault(key, []).append(r)

        steps: List[Dict[str, Any]] = []
        for key, group in groups.items():
            group = sorted(group, key=lambda r: r[target])
            for a, b in zip(group, group[1:]):
                if not a.get("cost") or not a.get("latency_s"):
                    continue
                d_cost = b["cost"] - a["cost"]
                steps.append({
                    f"from_{target}": a[target],
                    f"to_{target}": b[target],
                    "fixed": {n: v for n, v in zip(others, key)},
                    "d_cost_pct": round(100.0 * d_cost / a["cost"], 2),
                    "d_latency_pct": round(
                        100.0 * (b["latency_s"] - a["latency_s"]) / a["latency_s"], 2),
                    "direction": "better" if d_cost < 0 else
                                 ("worse" if d_cost > 0 else "flat"),
                })
        per_axis[target] = steps

    global_ranked = sorted(rows, key=lambda r: r.get("cost") or 1e9)[:10]

    return {
        "tool": "trend",
        "n_points": len(rows),
        "per_axis": per_axis,
        "global_top10": [
            {"point": {n: r[n] for n in names}, "cost": r.get("cost")}
            for r in global_ranked
        ],
        "note": ("per_axis lists are empty when no two sampled points share the "
                 "same values on the other dimensions. That is expected with "
                 "sparse sampling; use global_top10 instead."),
    }
```

### 7.4 `tool_validate` —— 收 point

```python
def tool_validate(store: Store, point: Dict[str, Any], space) -> Dict[str, Any]:
    errors: List[str] = []
    if not space.is_legal(point):
        errors.append(
            f"point {point} is not legal. Legal values per dimension: "
            + ", ".join(f"{n}={space.values(n)}" for n in space.names)
        )
    else:
        fp = space.fingerprint(point)
        if store.has_point(fp):
            errors.append(f"point {space.point_brief(point)} has already been simulated")

    legal_preview = [space.point_brief(p) for p in space.all_points()[:8]]
    return {
        "tool": "validate",
        "ok": not errors,
        "errors": errors,
        "legal_dimensions": {n: space.values(n) for n in space.names},
        "legal_preview": legal_preview + ["..."] if len(space.all_points()) > 8 else legal_preview,
    }
```

### 7.5 `TOOL_SPECS` 和 `dispatch`

```python
TOOL_SPECS = [
    {
        "name": "lookup_trials",
        "description": "List all simulated points, ranked by cost (lower is better).",
        "arguments": {"top_k": "optional int, default 20"},
    },
    {
        "name": "trend",
        "description": "Per-dimension first differences: along each axis, does cost still improve?",
        "arguments": {},
    },
    {
        "name": "inspect_point",
        "description": "Return the full record of one already-simulated point.",
        "arguments": {"fingerprint": "str, exactly as given in history"},
    },
    {
        "name": "validate_proposal",
        "description": "Check whether a proposed point is legal and not yet simulated. Call before proposing.",
        "arguments": {"point": "dict of all dimension values"},
    },
]


def dispatch(store: Store, name: str, args: Dict[str, Any], space) -> Dict[str, Any]:
    if name == "lookup_trials":
        return tool_lookup(store, space, int(args.get("top_k", 20)))
    if name == "trend":
        return tool_trend(store, space)
    if name == "inspect_point":
        return tool_inspect(store, str(args["fingerprint"]))
    if name == "validate_proposal":
        return tool_validate(store, args["point"], space)
    raise ValueError(f"unknown tool: {name}")
```

**验证**：需要一个有数据的库才能测。

```bash
python -c "
import json
from agent_dse.store import Store
from agent_dse.dims import Space
from agent_dse import tools
sp = Space(json.load(open('config/space.json')))
s = Store()
print('all_trials:', len(s.all_trials()))
t = tools.tool_trend(s, sp)
print('per_axis keys:', list(t['per_axis']))
print('global top1 :', t['global_top10'][0] if t['global_top10'] else None)
print('validate    :', tools.tool_validate(s, sp.baseline(), sp))
"
```

---

## 8. 第七步：改 `agent_dse/prompts.py`

### 为什么要改

现在的文案把一维写死了：

```
You are a hardware design-space-exploration analyst.
You read simulator results and identify where the bottleneck is and whether
adding DRAM bandwidth is still worthwhile.
```

LLM 会继续按"带宽"思考，即使你给了三个维度。而且 `PROPOSE_USER` 要求返回 `{"bw_gbps": int}`，必须改成返回完整 point。

### 新文案

```python
SYSTEM_DIAGNOSE = """\
You are a hardware design-space-exploration analyst.
You read simulator results and identify where the bottleneck is and which
direction of the design space is still worth exploring.

Rules:
- Only use numbers that appear in the data provided. Never invent values.
- `cost` is (latency/base_latency) * (energy/base_energy) * (area/base_area); LOWER IS BETTER.
- The design space has MULTIPLE tunable dimensions. Do not assume any single
  one is dominant; compare points on all dimensions.
- `area` grows with matrix size, vector width and DRAM channel count, so `cost`
  has an INTERIOR minimum: scaling hardware up buys latency but costs area.
- `dram_util` near 1.0 means DRAM is saturated. `matrix_util` low while
  `dram_util` high means the design is memory-bound; the opposite suggests the
  matrix array is the limiter.
- `dram_stall_cycles` is how long compute units waited on DRAM. If it is
  already near zero, more bandwidth cannot help.
- Output strict JSON, no prose, no markdown fences.
"""

DIAGNOSE_USER = """\
## Design space
{space}

## All simulated points
{trials}

## Per-dimension trend
{trend}

## Current best
{best}

Produce JSON:
{{
  "bottleneck": "memory_bound | compute_bound | balanced | unknown",
  "limiting_dimension": "<which dimension is most worth changing next, or 'unknown'>",
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
- `still_improving` = whether cost is still going DOWN along any axis inside the
  region you have already sampled. Crossing the interior minimum flips this to
  false, which is the normal expected outcome and NOT itself a reason to stop.
"""


SYSTEM_PROPOSE = """\
You choose the NEXT hardware point to simulate for a DSE loop.
You are given the diagnosis, the full history, the design space, and which
combinations are still untried.

Rules:
- The chosen point MUST have every dimension present, and every value MUST be
  one of the legal values for that dimension, and the combination MUST NOT
  have been simulated.
- Prefer interpolating inside a gap where the trend suggests the optimum lies.
- If the diagnosis says the curve has flattened, choose a nearby point to
  confirm the knee rather than jumping far away.
- Output strict JSON, no prose, no markdown fences.
"""

PROPOSE_USER = """\
## Design space
{space}

## Diagnosis
{diagnosis}

## Already simulated
{history}

## Untried combinations (first {untried_n} of {untried_total})
{untried}

## Best so far
{best}

Produce JSON:
{{
  "point": {{<all dimension names>: <int value>, ...}},
  "reason": "<why this point, referencing the numbers>",
  "expected_direction": "lower_cost | higher_cost | uncertain",
  "confidence": "low | medium | high"
}}
"""
```

**关键点**：`PROPOSE_USER` 里加了 `{space}`，让 LLM 每轮都能看到合法的维度名和候选值，而不是靠 system prompt 记忆。加第 4 维时这行自动生效。

---

## 9. 第八步：改 `agent_dse/agent.py`

改动集中在 `__init__`、`_untried`、`_space_brief`、`_history_brief`、`propose`。

```python
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from . import tools
from .dims import Space
from .llm import LLM
from .prompts import DIAGNOSE_USER, PROPOSE_USER, SYSTEM_DIAGNOSE, SYSTEM_PROPOSE
from .store import Store


class Agent:
    def __init__(self, llm: LLM, store: Store, space: Space) -> None:
        self.llm = llm
        self.store = store
        self.space = space

    # ---------------- 内部 ----------------
    def _untried(self) -> List[Dict[str, Any]]:
        return [
            p for p in self.space.all_points()
            if not self.store.has_point(self.space.fingerprint(p))
        ]

    def _space_brief(self) -> str:
        return self.space.describe()

    def _history_brief(self) -> str:
        rows = self.store.ok_trials()
        if not rows:
            return "(empty)"
        rows = sorted(rows, key=lambda r: r.get("cost") or 1e9)
        lines = []
        for r in rows:
            head = "  " + self.space.point_brief(r)
            lines.append(
                f"{head}  cost={r.get('cost'):.4f}  "
                f"latency={r.get('latency_s'):.6e}  energy={r.get('energy_j'):.6e}  "
                f"dram_util={r.get('dram_util')}  matrix_util={r.get('matrix_util')}  "
                f"dram_stall={r.get('dram_stall_cycles')}"
            )
        return "\n".join(lines)

    # ---------------- 第一段：诊断 ----------------
    def diagnose(self) -> Dict[str, Any]:
        trend = tools.tool_trend(self.store, self.space)
        user = DIAGNOSE_USER.format(
            space=self._space_brief(),
            trials=self._history_brief(),
            trend=json.dumps(trend["per_axis"], indent=2, ensure_ascii=False),
            best=json.dumps(self.store.best(), indent=2, ensure_ascii=False),
        )
        result = self.llm.chat_json(SYSTEM_DIAGNOSE, user)
        if not result.get("evidence"):
            raise ValueError(f"diagnosis has no evidence: {result}")
        return result

    # ---------------- 第二段：提案 ----------------
    def propose(self, diagnosis: Dict[str, Any], rnd: int) -> Dict[str, Any]:
        untried = self._untried()
        if not untried:
            return {"stop": True, "reason": "every legal combination has been simulated"}

        user = PROPOSE_USER.format(
            space=self._space_brief(),
            diagnosis=json.dumps(diagnosis, indent=2, ensure_ascii=False),
            history=self._history_brief(),
            untried_n=min(30, len(untried)),
            untried_total=len(untried),
            untried="\n".join(
                "  " + self.space.point_brief(p) for p in untried[:30]
            ),
            best=json.dumps(self.store.best(), indent=2, ensure_ascii=False),
        )

        last_error: Optional[str] = None
        for attempt in range(3):
            user_with_error = user if not last_error else (
                user + f"\n\n## Your previous answer was rejected\n{last_error}\n"
            )
            result = self.llm.chat_json(SYSTEM_PROPOSE, user_with_error)

            if result.get("stop") is True:
                return {"stop": True,
                        "reason": result.get("reason", "agent chose to stop")}

            # 1) 取出 point 并逐个维度转 int
            raw = result.get("point")
            if not isinstance(raw, dict):
                last_error = f"'point' must be a dict of dimension values, got {raw!r}"
                continue
            point: Dict[str, Any] = {}
            bad = None
            for name in self.space.names:
                if name not in raw:
                    bad = f"missing dimension '{name}'"
                    break
                try:
                    point[name] = int(raw[name])
                except (TypeError, ValueError):
                    bad = f"dimension '{name}' must be an int, got {raw[name]!r}"
                    break
            if bad:
                last_error = bad
                continue

            # 2) 合法性 + 去重
            check = tools.tool_validate(self.store, point, self.space)
            if not check["ok"]:
                last_error = "; ".join(check["errors"])
                continue

            # 3) 落库
            fp = self.space.fingerprint(point)
            self.store.put_decision(
                rnd, point, fp,
                reason=str(result.get("reason", "")),
                evidence=json.dumps(diagnosis.get("evidence", []), ensure_ascii=False),
            )
            return {"stop": False, "point": point, "fingerprint": fp, **result}

        raise ValueError(
            f"agent failed to produce a legal proposal in 3 attempts: {last_error}"
        )
```

### 三个必须注意的地方

1. **`int(raw[name])` 逐个维度转**，不是整体转。LLM 经常把数字写成字符串 `"4096"`。
2. **`point` 只保留搜索维度**，不把 `channel`/`density_gb` 交给 LLM（它们是 `bw_gbps` 的从属参数，由 `space.json` 的档预先绑定）。合法性检查只需要维度值。
3. **`return` 里同时给 `point` 和 `fingerprint`**，主循环拿哪个都方便。

---

## 10. 第九步：新增 `tests/test_08_full_loop_multidim.py`

这是全流程脚本，结构和 `test_06` 一样，只把"1 维选择"换成"点选择"。

```python
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_dse.agent import Agent
from agent_dse.costfn import cost as cost_fn
from agent_dse.dims import Space
from agent_dse.llm import LLM
from agent_dse.path import ARTIFACT_DIR, CONFIG_DIR, DATA_DIR, KEY_JSON_PATH, LOG_DIR
from agent_dse.sim import run_once
from agent_dse.store import Store

MAX_ITER = 10               # ← Agent 迭代轮数，你自己控制
MARGINAL_THRESHOLD_PER = 3
PATIENCE = 2


def pick_seeds(space: Space, n: int, strategy: str = "farthest"):
    """选 n 个点做初始种子。

    farthest: 从 baseline 出发，每次选"离已选集合最远"的点（归一化坐标）。
              确定性、覆盖均匀、不用随机数，适合演示。
    """
    points = space.all_points()
    base = space.baseline()
    if n >= len(points):
        return points

    def norm(p):
        v = []
        for name in space.names:
            vals = space.values(name)
            lo, hi = min(vals), max(vals)
            v.append(0.0 if hi == lo else (p[name] - lo) / (hi - lo))
        return tuple(v)

    if strategy == "random":
        import random
        rng = random.Random(0)
        chosen = [base]
        rest = [p for p in points if space.fingerprint(p) != space.fingerprint(base)]
        rng.shuffle(rest)
        return chosen + rest[: n - 1]

    chosen = [base]
    cand = [p for p in points if space.fingerprint(p) != space.fingerprint(base)]
    while len(chosen) < n and cand and len(chosen) < len(points):
        best_p, best_d = None, -1.0
        for p in cand:
            np_ = norm(p)
            d = min(
                sum((a - b) ** 2 for a, b in zip(np_, norm(c))) for c in chosen
            )
            if d > best_d:
                best_d, best_p = d, p
        chosen.append(best_p)
        cand.remove(best_p)
    return chosen


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    trace_path = LOG_DIR / "loop_trace_multidim.jsonl"
    if trace_path.exists():
        trace_path.unlink()

    run_cfg = json.loads((CONFIG_DIR / "run.json").read_text())
    space = Space(json.loads((CONFIG_DIR / "space.json").read_text()))
    all_points = space.all_points()

    store = Store(DATA_DIR / "trials.db")
    print(f"start  dims={space.names}  total={len(all_points)}  "
          f"existing={len(store.ok_trials())}", flush=True)

    # ---------- 阶段 A：种子点 ----------
    base_point = space.baseline()
    base_fp = space.fingerprint(base_point)
    need_base = not store.has_point(base_fp)

    if not store.ok_trials():
        seeds = pick_seeds(space, int(space.cfg.get("seed_n", 9)),
                           space.cfg.get("seed_strategy", "farthest"))
        if need_base:
            seeds = [base_point] + [p for p in seeds
                                    if space.fingerprint(p) != base_fp]
    else:
        seeds = []

    for i, p in enumerate(seeds, 1):
        fp = space.fingerprint(p)
        if store.has_point(fp):
            continue
        print(f"[seed {i}/{len(seeds)}] {space.point_brief(p)}", flush=True)
        t0 = time.time()
        try:
            rec = run_once(p, run_cfg, space)
            rec["status"] = "ok"
        except Exception as exc:
            rec = {**p, "fingerprint": fp, "status": "sim_error",
                   "error": str(exc), "elapsed_s": round(time.time() - t0, 1)}
        store.put_trial(rec)
        print(f"          -> {rec['status']} ({rec.get('elapsed_s')}s)", flush=True)

    # ---------- 阶段 B：归一化 + 播种 best ----------
    rows = store.ok_trials()
    if not rows:
        raise SystemExit("no successful trials; cannot compute baseline")

    base_rec = next((r for r in rows if r["fingerprint"] == base_fp), None)
    if base_rec is None:
        raise SystemExit(
            f"baseline point {space.point_brief(base_point)} was not simulated; "
            "it is required for cost normalisation"
        )
    base_metrics = {"latency_s": base_rec["latency_s"],
                    "energy_j": base_rec["energy_j"]}

    for r in store.ok_trials():
        r["cost"] = round(
            cost_fn(r, r, base_point, base_metrics, space.core_num), 6
        )
        store.conn.execute(
            "UPDATE trials SET cost=?, record=? WHERE fingerprint=?",
            (r["cost"], json.dumps(r), r["fingerprint"]),
        )
    store.conn.commit()
    for r in store.ok_trials():
        store.update_best(r, r["fingerprint"], r["cost"])

    print(f"baseline {space.point_brief(base_point)} -> cost = 1.0")
    print(f"best after seed: {store.best()}", flush=True)

    # ---------- 阶段 C：Agent 迭代 ----------
    agent = Agent(llm=LLM.from_file(KEY_JSON_PATH), store=store, space=space)
    stop_reason = "reached iteration limit."
    last_best_cost = None
    stale_rounds = 0

    for rnd in range(1, MAX_ITER + 1):
        print(f"\n{'=' * 60}\n[round {rnd}]", flush=True)

        best_now = store.best()
        marginal = None
        if last_best_cost is not None and best_now is not None:
            marginal = round(
                100 * (last_best_cost - best_now["cost"]) / last_best_cost, 3
            )
        last_best_cost = best_now["cost"] if best_now else None

        if marginal is not None and marginal <= MARGINAL_THRESHOLD_PER:
            stale_rounds += 1
        else:
            stale_rounds = 0

        if stale_rounds >= PATIENCE:
            stop_reason = (
                f"early stop: {stale_rounds} consecutive rounds without significant "
                f"improvement (last marginal = {marginal}% <= {MARGINAL_THRESHOLD_PER}%)"
            )
            print(stop_reason, flush=True)
            with trace_path.open("a") as f:
                f.write(json.dumps({"round": rnd, "event": "early_stop",
                                    "marginal_gain_pct": marginal,
                                    "stale_rounds": stale_rounds,
                                    "stop_reason": stop_reason,
                                    "best": best_now}) + "\n")
            break

        diagnosis = agent.diagnose()
        diagnosis["marginal_gain_pct_code"] = marginal
        print("diagnosis:", json.dumps(diagnosis, ensure_ascii=False), flush=True)

        choice = agent.propose(diagnosis, rnd)
        if choice.get("stop"):
            stop_reason = f"agent stopped: {choice.get('reason')}"
            print(stop_reason, flush=True)
            break

        p = choice["point"]
        print(f"agent chose: {space.point_brief(p)}", flush=True)
        print(f"  reason: {choice.get('reason')}", flush=True)

        t0 = time.time()
        try:
            rec = run_once(p, run_cfg, space)
            rec["status"] = "ok"
        except Exception as exc:
            rec = {**p, "fingerprint": space.fingerprint(p),
                   "status": "sim_error", "error": str(exc),
                   "elapsed_s": round(time.time() - t0, 1)}

        updated = "none"
        if rec["status"] == "ok":
            rec["cost"] = round(
                cost_fn(rec, rec, base_point, base_metrics, space.core_num), 6
            )
            store.put_trial(rec)
            updated = store.update_best(rec, rec["fingerprint"], rec["cost"])

        best = store.best()
        if rec["status"] == "ok":
            print(
                f"  {space.point_brief(rec)}  "
                f"L={rec['latency_s']:.4e}s  E={rec['energy_j']:.6f}J  "
                f"cost={rec['cost']:.4f}  stall={rec.get('dram_stall_cycles')}  "
                f"best={best['cost']:.4f} ({updated})",
                flush=True,
            )
        else:
            print(f"  status={rec['status']} error={rec.get('error', '')[:120]}",
                  flush=True)

        with trace_path.open("a") as f:
            f.write(json.dumps({
                "round": rnd,
                "diagnosis": diagnosis,
                "proposal": choice,
                "result": {k: rec.get(k) for k in (
                    "fingerprint", "status", "cost", "latency_s", "energy_j",
                    "dram_util", "matrix_util", "dram_stall_cycles", "elapsed_s")},
                "point": {n: rec.get(n) for n in space.names},
                "best_after": best,
                "best_update": updated,
            }, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 60}")
    print(f"stop_reason : {stop_reason}")
    print(f"simulated   : {len(store.ok_trials())} / {len(all_points)} legal points")
    print(f"final best  : {store.best()}")
    print(f"trace       : {trace_path}")
    print("STEP8_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### 与 `test_06` 的差异（只有 4 处）

| 差异 | 原因 |
|---|---|
| `pick_seeds()` 替代 `initial_offsets` | 多维下"下标"没有意义 |
| `space` 是 `Space` 对象而非 dict | 维度逻辑集中管理 |
| `run_once(p, run_cfg, space)` 替代 5 个位置参数 | 一个点就是一个 dict |
| `store.update_best(rec, fp, cost)` 替代 `(bw, cost)` | 需要指纹定位点 |

`MAX_ITER` / `PATIENCE` / `MARGINAL_THRESHOLD_PER` **完全保留**，你说"n 自己控制"就是改 `MAX_ITER`。

### 验证

```bash
rm -f data/trials.db                       # schema 变了，必须删
conda activate beacon
python tests/test_08_full_loop_multidim.py
```

期望看到：

```
start  dims=['bw_gbps', 'mac_num', 'vec_num']  total=120  existing=0
[seed 1/27] bw_gbps=1024 mac_num=6144 vec_num=512
          -> ok (35.2s)
...
baseline bw_gbps=1024 mac_num=6144 vec_num=512 -> cost = 1.0
[round 1]
diagnosis: {"bottleneck": "...", "limiting_dimension": "mac_num", ...}
agent chose: bw_gbps=2048 mac_num=8192 vec_num=512
  reason: ...
  bw_gbps=2048 mac_num=8192 vec_num=512  L=... E=... cost=... best=... (updated)
...
STEP8_OK
```

**耗时估算**：27 个种子 + 10 轮 ≈ 37 点 × ~35 s ≈ **22 分钟**。

如果只想快速验证链路通不通，把 `space.json` 的 `seed_n` 改成 `3`、`MAX_ITER` 改成 `2`，约 3 分钟。

---

## 11. 怎么加第 4 个维度（验证接口留对了）

假设你以后想加 `sram_kb`（每核 buffer 大小）。需要动的只有 **2 个文件**：

### 第 1 步：`config/space.json` 加一段

```json
{
    "name": "sram_kb",
    "levels": [
        {"buffer_size": 2048},
        {"buffer_size": 3072},
        {"buffer_size": 4096},
        {"buffer_size": 6144}
    ]
}
```

> 注意这里维度名 `sram_kb` 和它携带的字段 `buffer_size` 可以不同名——这正是"每档是一个 dict"的好处。
> `AREA_PER_UNIT` 里要登记名字 `sram_kb`，但 `sim.py` 需要的是 `buffer_size`。
> 所以 `dims.py` 的 `values()` 会去 `lvl["sram_kb"]` 取值，而 point 里会同时有 `sram_kb` 和 `buffer_size` 两个键。
> **如果嫌麻烦，最省事的做法是让维度名和字段名一致**，例如 `buffer_size` 直接作为维度名。

### 第 2 步：`costfn.py` 加一行（如果它消耗面积）

```python
AREA_PER_UNIT["buffer_size"]  = 0.0038    # mm^2 / KB
POWER_PER_UNIT["buffer_size"] = 0.0011    # W / KB
```

**完。**

`dims.py`、`store.py`、`tools.py`、`agent.py`、`prompts.py`、`sim.py`、测试脚本**全部不用改**：

- `fingerprint()` / `tag()` 自动多一段
- `describe()` 自动多一行
- prompt 里的 `{space}` 自动多一个维度
- LLM 返回的 `point` 自动多一个键
- `is_legal()` 自动检查这一维

### 唯一可能需要手改的地方

如果新维度**需要改 chip.yaml 里 ATLAS 没见过的字段**，才要在 `sim.py` 的 `build_system_yaml` 里加一行赋值。现有 5 个硬件字段（`mac_num`/`vec_num`/`buffer_size`/`buffer_read_bw`/`buffer_write_bw`）已经覆盖，加维度不用动。

---

## 12. 完整验证清单

按顺序做，每步都要过：

| # | 检查项 | 命令 | 期望 |
|---|---|---|---|
| 1 | `space.json` 可解析、维度数对 | `python -c "import json;print(len(json.load(open('config/space.json'))['dims']))"` | `3` |
| 2 | `Space` 能生成全部点 | `python -c "...print(len(sp.all_points()))"` | `80` |
| 3 | baseline 是各维中间档 | 同上，`sp.point_brief(sp.baseline())` | `bw_gbps=1024 mac_num=6144 vec_num=512` |
| 4 | fingerprint 唯一 | `len({sp.fingerprint(p) for p in sp.all_points()})` | `80` |
| 5 | 面积公式能复现 ATLAS preset | 见 0.5 节验算 | `6528 → 11.1566` |
| 6 | `run_once` 单点能跑通 | 见 Step 4 验证 | 打印 latency/energy |
| 7 | `store` 去重生效 | 见 Step 5 验证 | `True / False / True` |
| 8 | `tools.tool_trend` 不崩 | 见 Step 6 验证 | `per_axis` 有 3 个 key |
| 9 | Agent 能产出合法点 | 跑 `test_08` 看 `agent chose:` | 打印出一个完整 point |
| 10 | 全流程 | `python tests/test_08_full_loop_multidim.py` | `STEP8_OK` |

### 必须同步修改的旧文件

以下文件调用了被改动的接口，**不修改会直接报错**：

| 文件 | 要改什么 |
|---|---|
| `tests/test_02_check_runonce.py` | `run_once(bw, ch, dens, cfg, tag)` → `run_once(point, cfg, space)` |
| `tests/test_03_checkseed.py` | 种子格式要带三个维度；或直接弃用，改由 `test_08` 的阶段 A 生成 |
| `tests/test_04_check_tools.py` | `tool_validate(store, bw, points)` → `tool_validate(store, point, space)` |
| `tests/test_05_check_agent.py` | `Agent(llm, store, space_dict)` → `Agent(llm, store, Space_obj)` |
| `tests/test_06_check_full_loop.py` | 被 `test_08` 取代，建议保留作为"一维对照实验" |
| `tests/test_07_plot.py` | 若读取 `trials.db`，列名 `bw_gbps` 已不在表里，需从 `record` JSON 取 |

> **建议**：`test_06`（一维）和 `test_08`（三维）都留着。
> 汇报时"一维 Agent 找到 1024 是最优"和"三维 Agent 在 80 点空间里找到 X"是两组互补证据。

---

## 13. 常见故障排查

| 症状 | 原因 | 处理 |
|---|---|---|
| 三个维度的结果**完全一样** | `tag` 没有包含全部维度 → tiling 缓存串味（事实 B） | 确认 `run_once` 用的是 `space.tag(point)`；`ls data/artifacts/` 应看到每点一个目录 |
| `mac_num` 变了但 `energy_j` 几乎不变 | `sim.py` 没缩放 `power`（事实 C） | 确认 `build_system_yaml` 里有 `costfn.unit_power(...)` 两行 |
| `sqlite3.IntegrityError` 不报但点少了 | `fingerprint` 生成重复 | `len({sp.fingerprint(p) for p in sp.all_points()})` 应等于总点数 |
| `KeyError: 'bw_gbps'` | 还有代码在按 bw 取名 | 全局 `grep -rn "bw_gbps" agent_dse/`，改成 `point[name]` |
| Agent 反复提案被拒 | LLM 选了已跑过的组合 | 正常，`validate` 在工作；看 `last_error`，通常是 `untried` 列表被截断到 30 条导致它看不到全部 |
| `per_axis` 全是空 list | 种子点太稀疏，没有"其余维度相同"的点对 | 正常，靠 `global_top10`；或把 `seed_n` 调大 |
| 仿真结果和旧版 `test_06` 不可比 | `power` 缩放改变了能量 | 这是**预期**的。新 cost 里面积项也变了，两组绝对值不可直接比，只比"是否找到内部最优" |
| `OSError: libramulator.so` | 用了 `atlas` 环境（Python 3.10） | 必须用 `beacon` 环境（Python 3.12），见原手册 Step 0 |

---

## 附：本手册改动的文件清单

| 文件 | 动作 |
|---|---|
| `config/space.json` | 重写（3 维定义） |
| `config/run.json` | 精简（移走 core_num / fixed_hardware） |
| `agent_dse/dims.py` | **新增**（维度引擎，扩展接口核心） |
| `agent_dse/costfn.py` | **新增**（面积/功耗模型） |
| `agent_dse/sim.py` | 改 3 处（power 缩放、point 入参、tag） |
| `agent_dse/store.py` | 改 schema + 5 个方法 |
| `agent_dse/tools.py` | 4 个工具改 point 语义 |
| `agent_dse/prompts.py` | 3 段文案重写 |
| `agent_dse/agent.py` | 改 5 个方法 |
| `tests/test_08_full_loop_multidim.py` | **新增** |
| `tests/test_02~07` | 同步小改（见第 12 节） |

**没有改动 ATLAS 一个字节。**
