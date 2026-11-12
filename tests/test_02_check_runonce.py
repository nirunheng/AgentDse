from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_dse.path import CONFIG_DIR, LOG_DIR
from agent_dse.sim import run_once

def main()->int:
    run_cfg = json.loads((CONFIG_DIR / "run.json").read_text())

    # 第一次跑设定带宽为1TB
    metrics = run_once(
        bw_gbps=1024,
        channel=16,
        density_gb=4,
        run_cfg=run_cfg,
        tag="one_1024GBps"
    )

    LOG_DIR.mkdir(exist_ok=True, parents=True)
    out = LOG_DIR / "check_runonce.json"
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