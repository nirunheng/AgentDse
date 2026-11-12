import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_dse import dramcfg
from agent_dse.agent import Agent
from agent_dse.store import Store
from agent_dse.llm import LLM
from agent_dse.path import DATA_DIR, CONFIG_DIR, LOG_DIR, KEY_JSON_PATH, ARTIFACT_DIR
from agent_dse.sim import run_once

MAX_ITER = 15
MARGINAL_THRESHOLD_PER = 3
# 需要连续多少轮“无显著改进”才真正早停。单轮无改进就停会过早收敛：
# 一维曲线上最优点的邻居往往只比它差一点点，Agent 探完一个坏邻居就会误停。
PATIENCE = 2
# area normalization reference; must be in space.json as baseline_channel
BASELINE_CHANNEL = 8 


def _cost(row, base, base_ch:int=BASELINE_CHANNEL):
    """(L/L0) * (E/E0) * (A/A0).  Area scales linearly with channel count,
    which is what creates an INTERIOR optimum (see manual §0.5 结论 C)."""
    return (
        (row["latency_s"] / base["latency_s"])
        * (row["energy_j"] / base["energy_j"])
        * (row["channel"] / base_ch)
    )

def channel_of(space, bw: int):
    for p in space["points"]:
        if p["channel"] * 64 == bw:
            return p
    raise ValueError(f"bw{bw} not in space")

def main()->int:
    LOG_DIR.mkdir(exist_ok=True, parents=True)
    trace_path = LOG_DIR / "loop_trace.jsonl"
    if trace_path.exists():
        trace_path.unlink()

    run_cfg = json.loads((CONFIG_DIR / "run.json").read_text())
    space = json.loads((CONFIG_DIR / "space.json").read_text())
    legal = sorted(p["channel"]*64 for p in space["points"])

    store = Store(DATA_DIR / "trials.db")
    # 稳健性：允许 data/dram_cfgs 与 artifacts 在多次运行间复用，
    # 但 trials.db 会保留历史。若你想从零重跑，先删 data/trials.db。

    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] start  db={DATA_DIR / 'trials.db'}"
          f"existing trials={len(store.ok_trials())}")

    # 阶段A 补齐initial_offsets补充的种子点的仿真
    if not store.ok_trials():
        for idx in space["initial_offsets"]:
            bw = legal[idx]   
            if store.has_bw(bw):
                continue
            p = channel_of(space, bw)
            print(f"[init] simulating bw={bw} GB/s ...", flush=True)
            t0 = time.time()
            try:
                rec = run_once(bw, p["channel"], p["density_gb"], run_cfg, tag=f"bw{bw}")
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

    # 阶段B 算Baseline迭代
    rows = store.ok_trials()
    if not rows:
        raise SystemExit("no successful trials; cannot compute baseline")

    base = next((r for r in rows if r["bw_gbps"] == space["baseline_bw_gbps"]), None)
    if base is None:
        base = min(rows, key=lambda r: abs(r["bw_gbps"] - space["baseline_bw_gbps"]))
        print(f"warning: baseline {space['baseline_bw_gbps']} mising, using{base['bw_gbps']} as baseline")

    base_ch = int(space.get("baseline_channel", BASELINE_CHANNEL))
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

    # ---阶段c： Agent迭代计算 ---
    agent = Agent(llm=LLM.from_file(KEY_JSON_PATH), store=store, space=space)
    stop_reason : str = "reached iteration limit."
    last_best_cost = None
    stale_rounds = 0          # 连续多少轮 best 没有显著改进

    for rnd in range(1, MAX_ITER+1):
        print(f"\n{'='*60}\n[round{rnd}]", flush=True)
        # 边际收益必须由代码算，不能信 LLM 自报的数字：
        # 实测 LLM 会把“cost 上涨 217%”当成 marginal_gain_pct 返回，导致误停。
        best_now = store.best()
        marginal = None
        if last_best_cost is not None and best_now is not None:
            marginal = round(
                100 * (last_best_cost - best_now["cost"]) / last_best_cost,
                3
            )
        last_best_cost = best_now["cost"] if best_now else None

        # 只在“本轮没有显著改进”时累加，有改进就清零重来。
        # 这样 Agent 可以连续探测多个候选点，而不是探一个坏点就停。
        if marginal is not None and marginal <= MARGINAL_THRESHOLD_PER:
            stale_rounds += 1
        else:
            stale_rounds = 0

        if stale_rounds >= PATIENCE:
            stop_reason = (
                f"early stop: {stale_rounds} consecutive rounds without a significant"
                f" improvement (last marginal gain = {marginal}% <= threshold"
                f" {MARGINAL_THRESHOLD_PER}%) (best={best_now})"
            )
            print(stop_reason, flush=True)
            with trace_path.open("a") as f:
                f.write(json.dumps({
                    "round":rnd,
                    "event":"early_stop",
                    "marginal_gain_pct": marginal,
                    "stale_rounds": stale_rounds,
                    "stop_reason":stop_reason,
                    "best":best_now
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
        print(f"agent chose bw={bw}GB/s :: {choice.get('reason')}", flush=True)
                
        t0 = time.time()
        try:
            rec = run_once(bw, p["channel"], p["density_gb"], run_cfg, tag=f"bw{bw}")
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

