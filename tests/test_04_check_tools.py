import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_dse.store import Store
from agent_dse.path import DATA_DIR, CONFIG_DIR
from agent_dse import tools

def main()->int:
    s = Store(DATA_DIR / "test.db")
    for p in sorted((DATA_DIR / "seed").glob("*.json")):
        s.put_trial(json.loads(p.read_text()))

    space = json.loads((CONFIG_DIR / "space.json").read_text())["points"]

    for r in s.ok_trials():
        s.update_best(r["bw_gbps"], r["cost"])

    print(json.dumps(tools.tool_lookup(s, 3), indent=2))
    print(json.dumps(tools.tool_trend(s)["steps"][-3:], indent=2))
    print(json.dumps(tools.tool_validate(s, 1536, space), indent=2))
    print("STEP5_OK")


if __name__ == "__main__":
    raise SystemExit(main())