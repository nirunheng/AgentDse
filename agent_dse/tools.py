from __future__ import annotations
from typing import Any, Optional, Dict, List
import json
from .store import Store


# tool1 看当前排行榜
def tool_lookup(store: Store, names: List[str], top_k: int=20)->Dict[str, Any]:
    rows = store.ok_trials()
    ranked = sorted(rows, key=lambda r: r["cost"])[:top_k]
    return {
        "tool": "lookup_trials",
        "n_total": len(store.all_trials()),
        "n_ok" : len(rows),
        "best": store.best(),
        "ranked": [_compact(r, names) for r in ranked]
    }

#tool2 看某一个具体点的完整观测
def tool_inspect(store: Store, fingerprint: str) -> Dict[str, Any]:
    rec = store.get_trial(fingerprint)
    if rec is None:
        return {"tool": "inspect_point", "found": False,
                "fingerprint": fingerprint,
                "note": "this point has not been simulated yet"}
    return {"tool": "inspect_point", "found": True, "point": rec}

#tool3 see the trend of result(一阶导数)
# def tool_trend(store: Store)->Dict[str, Any]:
#     rows = sorted(store.ok_trials(), key=lambda r: r["bw_gbps"])
#     steps: List[Dict[str, Any]] = []
#     for a, b in zip(rows, rows[1:]):
#         d_cost = b["cost"] - a["cost"]
#         steps.append({
#             "from_bw": a["bw_gbps"],
#             "to_bw": b["bw_gbps"],
#             "d_cost": round(d_cost, 4),
#             "d_cost_pct": round(100.0 * d_cost / a["cost"], 2) if a["cost"] else None,
#             "d_latency_pct": round(100.0 * (b["latency_s"] - a["latency_s"]) / a["latency_s"], 2),
#             "d_energy_pct": round(100.0 * (b["energy_j"] - a["energy_j"]) / a["energy_j"], 2),
#             "direction": "better" if d_cost < 0 else ("worse" if d_cost > 0 else "flat")
#         })
    
#     return {
#         "tools":"trend",
#         "n_points":len(rows),
#         "steps": steps
#     }

#new tool3 trend
def tool_trend(store:Store, space)->Dict[str, Any]:
    """"
    对于每个点，找出它的除了该维度其他都相同，且该维度相邻的点做一阶差分来判断趋势
    三维空间里这种同profile组常常只有2个点或者一个点，所以额外给一个global端，按cost排序的前若干点，供LLM直接比较
    """

    rows = store.ok_trials()
    names = space.names
    per_axis: Dict[str, List[Dict[str, Any]]] = {}

    for target in names:
        others =  [n for n in names if n!= target]
        groups:Dict[tuple, List[Dict[str, Any]]] = {}
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


#Tool4 check whether a proposal is leggal
def tool_validate(store:Store, point:Dict[str, Any], space)->Dict[str, Any]:
    errors:List[str]=[]
    if not space.is_legal(point):
        errors.append(
            f"point {point} is not legal. Legal values oer dimension："
            + ",".join(f"{n}={space.values(n)}" for n in space.names)
        )

    else:
        fp = space.fingerprint(point)
        if store.has_point(fp):
            errors.append(f"point {space.point_brief(point)} has already been simulated")

    legal_preview = [space.point_brief(p) for p in space.all_points()[:8]]
    return {
        "tool": "validate",
        "ok" : not errors,
        "error" : errors,
        "legal_dimensions":{n:space.values(n) for n in space.names},
        "legal_preview": legal_preview + ["..."] if len(space.all_points()) >8 else legal_preview
    }

def _compact(r: Dict[str, Any], names:List[str]) -> Dict[str, Any]:

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

