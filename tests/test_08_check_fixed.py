import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent_dse.sim import run_once
from agent_dse.dims import Space

def main()->int:
    sp = Space(json.load(open('config/space.json')))
    run_cfg = json.load(open('config/run.json'))
    p = sp.baseline()
    rec = run_once(p, run_cfg, sp, tag='smoke_baseline')
    print({k: rec[k] for k in ('bw_gbps','latency_s','energy_j','channel')})
    print('fingerprint =', rec['fingerprint'])


if __name__ == "__main__":
    main()
