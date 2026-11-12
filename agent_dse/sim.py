from __future__ import annotations
import json
import os
import shutil
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional

import yaml
from .import dramcfg
from .path import DRAM_DIR, ARTIFACT_DIR, ATLAS, ATLAS_LIB, ROOT

WORKER = Path(__file__).resolve().parent / "sim_worker.py"

#芯片的模板JSON
_CHIP_TEMPLATE = {
    "architecture":{
        "frequency": 1000,
        "core_num":16,
        "core":{
            "controller":{
                "power":1.0,
                "area":1.3
                },
            "matrix":{
                "mac_num":8192,
                "power":1.1,
                "area":1.1
            },
            "vector":{
                "vec_num":512,
                "power":1.2,
                "area":1.2
            },
            "buffer":{
                "buffer_size":3072,
                "read_bw":8192.0,
                "write_bw":8192.0,
                "power":1.3,
                "area":1.3
            }
        },
        "dram":{
            "config_path":"",
            "power":1.4,
            "area":1.4
        },
        "noc":{
            "topology":"mesh",
            "config_path":"configs/architecture/noc/4x4_mesh",
            "flit_size":64,
            "power":1.5,
            "area":1.5
        }
    }
}

#system模板JSON
_SYSTEM_TEMPLATE = {
    "system":{
        "type":"cloud",
        "chip":{
            "config_path":"",
        },
        "parallelism":{
            "tp_size":8,
            "ep_size":8,
            "tp_scope":"scale_up",
            "ep_scope":"scale_up"
        },
        "interconnect":{
            "scale_up_latency":4e-7,
            "scale_up_bandwidth":450,
            "scale_up_energy":1.3,
            "scale_out_latency":2e-6,
            "scale_out_bandwidth":50,
            "scale_out_energy":1.3
        },
        "kv_cache":{
            "max_context_length":532488,
            "block_size":256,
        }
    }
}

# 这一步是在构建一个完整的system_yaml文件，返回system_yaml路径
def build_system_yaml(
    out_dir:Path,
    dram_yaml:Path,
    fixed_hardware: Dict[str, Any],
    core_num:int=16
)->Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    chip = json.loads(json.dumps(_CHIP_TEMPLATE))
    arch = chip["architecture"]
    arch["core_num"] = core_num
    arch["dram"]["config_path"] = str(dram_yaml)
    arch["frequency"]=float(fixed_hardware["frequency"])
    arch["core"]["matrix"]["mac_num"] = int(fixed_hardware["mac_num"])
    arch["core"]["vector"]["vec_num"] = int(fixed_hardware["vec_num"])
    arch["core"]["buffer"]["buffer_size"] = int(fixed_hardware["buffer_size"])
    arch["core"]["buffer"]["read_bw"] = float(fixed_hardware["buffer_read_bw"])
    arch["core"]["buffer"]["write_bw"] = float(fixed_hardware["buffer_write_bw"])

    chip_path = out_dir / "chip.yaml"
    with chip_path.open("w") as f:
        yaml.safe_dump(chip, f, sort_keys=False)

    system = json.loads(json.dumps(_SYSTEM_TEMPLATE))
    system["system"]["chip"]["config_path"] = str(chip_path.resolve())
    system_path = out_dir / "system.yaml"
    with system_path.open("w") as f:
        yaml.safe_dump(system, f, sort_keys=False)

    return system_path

def run_once(
    bw_gbps: int,
    channel: int,
    density_gb: int,
    run_cfg: Dict[str, Any],
    tag: str,   
)->Dict[str, Any]:
    # first step: generate Dram_Config
    dram_yaml = dramcfg.write_cfg(channel=channel, density_gb=density_gb)
    out_dir = ARTIFACT_DIR / tag
    if out_dir.exists():
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    system_yaml = build_system_yaml(
        out_dir, dram_yaml=dram_yaml, fixed_hardware=run_cfg["fixed_hardware"], core_num=run_cfg["core_num"]
    )

    spec_path = out_dir / "spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "system_yaml":str(system_yaml),
                "model_preset": run_cfg["model_preset"],
                "batch_size": run_cfg["batch_size"],
                "context_length":run_cfg["context_length"],
                "element_size": run_cfg["element_size"]
            }
        )
    )

    result_path = out_dir / "metrics.json"
    artifacts = out_dir / "atlas_artifacts"
    artifacts.mkdir(exist_ok=True, parents=True)

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{ATLAS_LIB}:" + env.get("LD_LIBRARY_PATH", "")
    env["PYTHONPATH"] = f"{ATLAS_LIB}:{ATLAS}:" + env.get("PYTHONPATH", "")
    env["AGENTDSE_TILING_WORKERS"] = str(run_cfg.get("num_workers", 4))

    log_path = out_dir / "atlas.log"
    with log_path.open("w") as log:
        proc = subprocess.run(
            [sys.executable, str(WORKER), str(spec_path), str(artifacts), str(result_path)],
            stdout=log,
            stderr=log,
            env=env,
            timeout=int(run_cfg.get("timeout_s", 900)),
        )

    if proc.returncode != 0 or not result_path.exists():
        raise RuntimeError(f"ATLAS failed (rc={proc.returncode}); see {log_path}")

    payload = json.loads(result_path.read_text())
    payload["bw_gbps"] = bw_gbps
    payload["channel"] = channel
    payload["density_gb"] = density_gb
    payload["run_dir"] = str(out_dir)
    return payload

