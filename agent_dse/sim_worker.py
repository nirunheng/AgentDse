from __future__ import annotations

import json
import os
import multiprocessing
import sys
import time
from pathlib import Path

_WORKERS = int(os.environ.get("AGENTDSE_TILING_WORKERS", "4"))
multiprocessing.cpu_count = lambda: max(2, int(_WORKERS / 0.8))

ATLAS_PATH = str(Path(__file__).resolve().parent / "ATLAS-MICRO-2026-main")

sys.path.insert(0, ATLAS_PATH + "/simulator/build/lib")
sys.path.insert(0, ATLAS_PATH)
os.chdir(ATLAS_PATH)

def main()->int:
    if len(sys.argv) != 4:
        raise ValueError(f"Usage: python ./sim_worker.py <system.json> <artifacts_dir> <out_json>")

    system_json_path = Path(sys.argv[1])
    artifacts_dir = Path(sys.argv[2])
    out_json_path = Path(sys.argv[3])

    import yaml
    from frontend.hardware_parser import CloudSystemConfig
    from frontend.model_parser import get_model_config_from_hf
    from frontend.util import generate_context_slot_mapping
    from frontend.auto.system import CloudSystem

    spec = json.loads(Path(system_json_path).read_text())

    with open(spec["system_yaml"]) as f:
        cc = CloudSystemConfig.from_yaml(yaml.safe_load(f))

    model_config = get_model_config_from_hf(
        spec["model_preset"],
        os.path.join(ATLAS_PATH, f"configs/models/{spec['model_preset']}.json"),
    )

    ctx = [int(spec["context_length"])] * int(spec["batch_size"])
    system = CloudSystem(
        system_config=cc,
        model_config=model_config,
        element_size=int(spec["element_size"]),
    )
    slot_map = generate_context_slot_mapping(cc, ctx)

    t0 = time.time()
    result = system.inference(
        context_length_list=ctx,
        context_slot_mapping=slot_map,
        intermediate_result_dir=artifacts_dir,
    )
    elapsed = time.time() - t0

    stats = result["intra_chip_computation_performance"].e2e_stats
    payload = {
        "latency_s": float(result["latency"]),
        "energy_j": float(result["energy"]),
        "matrix_util": float(stats.matrix_util()),
        "vector_util": float(stats.vector_util()),
        "dram_util": float(stats.dram_util()),
        "buffer_util": float(stats.buffer_util()),
        "noc_util": float(stats.noc_util()),
        "dram_stall_cycles": int(stats.matrix_bubble_dram_cycles),
        "onchip_stall_cycles": int(stats.matrix_bubble_on_chip_cycles),
        "dram_cycles": int(stats.dram_cycles),
        "matrix_cycles": int(stats.matrix_cycles),
        "dram_energy_j": float(stats.dram_energy),
        "matrix_energy_j": float(stats.matrix_energy),
        "vector_energy_j": float(stats.vector_energy),
        "buffer_energy_j": float(stats.buffer_energy),
        "noc_energy_j": float(stats.noc_energy),
        "memory_access_bytes": float(stats.memory_access_bytes),
        "flop_count": float(stats.flop_count),
        "arithmetic_intensity": float(stats.arithmetic_intensity()),
        "elapsed_s": round(elapsed, 2),
    }

    Path(out_json_path).write_text(json.dumps(payload, indent=2))
    print(json.dumps({"ok": True, "elapsed_s": payload["elapsed_s"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


