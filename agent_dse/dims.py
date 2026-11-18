from __future__ import annotations

from itertools import product
from typing import Any, Optional, Dict, List

class Space:
    """维度无关的搜索空间，读取config里的结构，不用提前知道任何维度名"""
    def __init__(self, cfg: Dict[str, Any])->None:
        self.cfg = cfg
        self.dims: List[Dict[str, Any]] = cfg["dims"]
        self.name: List[str] = [d["name"] for d in cfg["dims"]]
        self.names = self.name
        self.core_num = int(cfg.get("core_num", 16))
        self.fixed_hardware:Dict[str, Any] = dict(cfg.get("fixed_hardware", {}))

    # 查询函数，查询当前name对应多少个point
    def levels(self, name:str)->List[Dict[str, Any]]:
        for d in self.dims:
            if d["name"] == name:
                return d["levels"]

        raise KeyError(f"unknown dimension:{name}")

    def values(self, name:str)-> List[int]:
        return [lvl[name] for lvl in self.levels(name)]

    #笛卡尔积
    def all_points(self)->List[Dict[str, Any]]:
        out:List[Dict[str, Any]] = []
        for combo in product(*[d["levels"] for d in self.dims]):
            point: Dict[str, Any] = {}
            for lvl in combo:
                point.update(lvl)
            point.update(self.fixed_hardware)
            out.append(point)

        return out

    # fingerprint是唯一sqlite名
    def fingerprint(self, point:Dict[str, Any])->str:
        return "|".join(f"{n}={point[n]}" for n in self.names)

    # artifact目录名，为避免缓存重复命中
    def tag(self, point:Dict[str, Any])->str:
        return "_".join(f"{n}{point[n]}" for n in self.names)

    # basline
    def baseline(self)->Dict[str, Any]:
        """选取原则:每个dim选中间值作为baseline"""
        point:Dict[str, Any] = {}
        for d in self.dims:
            lvl = d["levels"][len(d["levels"]) // 2]
            point.update(lvl)
        point.update(self.fixed_hardware)
        return point

    # check whether a point is legal
    def is_legal(self, point:Dict[str, Any])->bool:
        if not isinstance(point, dict):
            return False

        for name in self.names:
            if name not in point:
                return False

            if point[name] not in self.values(name):
                return False

        return True

    # 给prompt用的文本
    def describe(self)->str:
        lines = [f"There are {len(self.names)} tunable dimensions "
                 f"({len(self.all_points())} legal combinations in total):"]
        for d in self.dims:
            val = self.values(d["name"])
            lines.append(f"  - {d['name']}: {val}")

        lines.append("")
        lines.append("Extra fixed fields carried by every point "
                     "(do NOT propose these, they are not tunable):")
        lines.append(f"  {self.fixed_hardware}")
        return "\n".join(lines)

    def point_brief(self, point: Dict[str, Any]) -> str:
        return " ".join(f"{n}={point[n]}" for n in self.names)
