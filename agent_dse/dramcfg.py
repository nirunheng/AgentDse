from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from .path import DRAM_DIR

TEMPLATE = {
    "Frontend": {
        "impl": "HBFrontend",
        "clock_ratio": 1,
    },
    "MemorySystem":{
        "impl":"GenericDRAM",
        "clock_ratio":1,
        "DRAM":{
            "impl":"HBDRAM",
            "org":{
                "preset":"HBDRAM_4Gb_1024pin_512col",
                "channel":16
            },
            "timing":{"preset":"HBDRAM_500Mbps"},
        },
        "Controller":{
            "impl":"Generic",
            "Scheduler":{"impl":"FRFCFS"},
            "RefreshManager":{"impl":"AllBank"},
            "RowPolicy":{"impl":"OpenRowPolicy"},
            "plugins":None,
        },
        "AddrMapper":{
            "impl":"OneLevelInterleave",
            "channel_lowest_bit":5,
        },
    },
}

LEGAL_DENSITY_GB = {1, 2, 4, 8, 16}

def bw_gbps(channel: int) ->int:
    return channel * 64

def make_cfg(channel: int, density_gb: int) ->Dict:
    if density_gb not in LEGAL_DENSITY_GB:
        raise ValueError(f"illegal density_gb={density_gb}, must be one of {LEGAL_DENSITY_GB}")

    if channel < 1 or (channel & (channel -1)) != 0:
        raise ValueError(f"illegal channel={channel}, must be a positive power of 2")

    cfg = yaml.safe_load(yaml.safe_dump(TEMPLATE))
    cfg["MemorySystem"]["DRAM"]["org"]["preset"] = f"HBDRAM_{density_gb}Gb_1024pin_512col"
    cfg["MemorySystem"]["DRAM"]["org"]["channel"] = int(channel)
    return cfg

def write_cfg(channel: int, density_gb: int)->Path:
    DRAM_DIR.mkdir(exist_ok=True, parents=True)
    path = DRAM_DIR / f"bw{bw_gbps(channel)}GBps.yaml"
    with path.open("w") as f:
        yaml.safe_dump(make_cfg(channel, density_gb), f, sort_keys=False)
    return path

def write_all(points: List[Dict])-> Dict[int, str]:
    """points: [{"channel": 4, "density_gb": 16}, ...] -> {bw_gbps: path_str}"""
    out: Dict[int, str] = {}
    for p in points:
        out[bw_gbps(p["channel"])] = str(write_cfg(p["channel"], p["density_gb"]))
    return out
