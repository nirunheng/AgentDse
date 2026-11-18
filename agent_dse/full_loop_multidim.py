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

from typing import Dict, Any, Optional, List

MAX_ITER = 20
MARGINAL_THRESHOLD_PER = 3
PATIENCE = 5

def pick_seeds(space:Space, n: int, strategy: str = "farthest") -> Dict[str, Any]:
    """
    这个函数用于选择n个点作为初始种子,从baseline开始每次选择“距离已选点集合最远的点”
    确保覆盖均匀，这次我们不用随机数
    """
    points = space.all_points()
    baseline = space.baseline()
    if n >= len(points):
        return points

    def norm(p):
        v = []
        for name in space.names:
            vals = space.values(name)
            lo, hi = min(vals), max(vals)
            v.append(0.0 if hi == lo else (p[name] - lo) / (hi - lo))
        return tuple(v)

    if strategy == "ramdom":
        import random
        rng = random.Random(0)
        chosen = [baseline]
        rest = [p for p in points if space.fingerprint(p) != space.fingerprint(baseline)]
        rng.shuffle(rest)
        return chosen + rest[: n-1]

    chosen = [baseline]
    cand = [p for p in points if space.fingerprint(p) != space.fingerprint(baseline)]
    while len(chosen) < n and cand and len(chosen) < len(points) :
        best_p, best_d = None, -1.0
        for p in cand:
            np = norm(p)
            d = min(
                sum((a-b) ** 2 for a,b in zip(np, norm(c))) for c in chosen
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
    print(f"start  dims={space.names}  total={len(all_points)} "
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



