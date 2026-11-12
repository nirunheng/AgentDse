"""Write the 7 measured points as seed trials.

Numbers are from real ATLAS e2e runs (llama3_8b, batch=8, ctx=4096,
core_num=16, HBDRAM_500Mbps). cost includes the DRAM area term
A/A0 = channel/8, so the optimum is interior (channel=16).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent_dse.path import CONFIG_DIR, DATA_DIR

# ALL measure
# channel, density_gb, Latency_s, energy_j, matrix_util, dram_util, dram_stall, e2e_cycle
MEASURED = {
    (2,   16, 1.461344e-3, 0.1754376163328, 0.0428, 0.9412, 39106, 44867),
    (4,   16, 8.25536e-4,  0.0991406563328, 0.0768, 0.8889, 19358, 24998),
    (8,    8, 4.92384e-4,  0.0591624163328, 0.1316, 0.7774,  9002, 14587),
    (16,   4, 3.34144e-4,  0.0401736163328, 0.1991, 0.6882,  4178,  9642),
    (32,   2, 2.60512e-4,  0.0313377763328, 0.2615, 0.5672,  1858,  7341),
    (64,   1, 2.29504e-4,  0.0276168163328, 0.3013, 0.5107,   932,  6372),
    (128,  1, 2.19264e-4,  0.0263880163328, 0.3173, 0.4881,   620,  6052)
}

def cost_of(row, base, base_area):
    return (
        (row["latency_s"] / base["latency_s"])
        * (row["energy_j"] / base["energy_j"])
        * (row["channel"] / base_area)
    )

def main()->int:
    space = json.loads((CONFIG_DIR / "space.json").read_text())
    out_dir = DATA_DIR / "seed"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for ch, den, lat, eng, multil, dutil, stall, cycles in MEASURED:
        rows.append({
            "channel":ch, "density_gb":den, "bw_gbps":ch*64, "latency_s":lat, 
            "energy_j":eng, "matrix_util":multil, "dram_util":dutil,
            "dram_stall_cycle":stall, "e2e_cycles":cycles,
            "status": "ok", "rundir": "SEED"
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


