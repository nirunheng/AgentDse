import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agent_dse.path import DATA_DIR, LOG_DIR

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