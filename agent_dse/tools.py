from __future__ import annotations
from typing import Any, Optional, Dict, List
import json
from .store import Store


# tool1 看当前排行榜
def tool_lookup(store: Store, top_k: int=20)->Dict[str, Any]:
    rows = store.ok_trials()
    ranked = sorted(rows, key=lambda r: r["cost"])[:top_k]
    return {
        "tool": "lookup_trials",
        "n_total": len(store.all_trials()),
        "n_ok" : len(rows),
        "best": store.best(),
        "ranked": [_compact(r) for r in ranked]
    }

#tool2 看某一个具体的带宽点的完整观测
def tool_inspect(store: Store, bw: int)->Dict[str, Any]:
    for r in store.all_trials():
        if r["bw_gbps"] == bw:
            return {
                "tool": "inspect_point",
                "found": True,
                "point": r
            }
    return {
        "tool": "inspect_point",
        "found":False,
        "bw_gbps": bw,
        "note": "this bandwidth has not been simulated yet"
    }

#tool3 see the trend of result(一阶导数)
def tool_trend(store: Store)->Dict[str, Any]:
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
            "direction": "better" if d_cost < 0 else ("worse" if d_cost > 0 else "flat")
        })
    
    return {
        "tools":"trend",
        "n_points":len(rows),
        "steps": steps
    }

#Tool4 check whether a proposal is leggal
def tool_validate(store:Store, bw:int, space_points: List[Dict[str, int]])->Dict[str, Any]:
    legal = {p["channel"] * 64 for p in space_points}
    errors: List[str] = []
    if bw not in legal:
        errors.append(
            f"bw={bw}GB/s is not in the design space: legal values are{sorted(legal)}"
        )
    if store.has_bw(bw):
        errors.append(
            f"bw={bw}GB/s has already been simulated; pick an untried value"
        )

    return{
        "tool":"validate",
        "ok": not errors,
        "errors": errors,
        "legal_values": sorted(legal)
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

