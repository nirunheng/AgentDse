from __future__ import annotations
from typing import Optional, Any, Dict, List

#以下系数全部从ATLAS自己的Preset推得，是DSE的重要依据
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

AREA_PER_UNIT :Dict[str, float] = {
    "mac_num": MAC_AREA_PER_UNIT,
    "vec_num": VEC_AREA_PER_UNIT
}

POWER_PER_UNIT: Dict[str, float] = {
    "mac_num": MAC_POWER_PER_UNIT,
    "vec_num": VEC_POWER_PER_UNIT
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