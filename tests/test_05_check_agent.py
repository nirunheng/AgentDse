import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_dse.agent import Agent
from agent_dse.llm import LLM
from agent_dse.path import CONFIG_DIR, DATA_DIR, KEY_JSON_PATH, LOG_DIR
from agent_dse.store import Store

def main()->int:
    db = DATA_DIR / "agent_test.db"
    if db.exists():
        db.unlink()
    store = Store(db)

    space = json.loads((CONFIG_DIR / "space.json").read_text())

    # 这里首先仿真事先想要的几个offsets的数值
    legal = sorted(p["channel"] * 64 for p in space["points"])
    wanted = {legal[i] for i in space["initial_offsets"]}

    for p in sorted((DATA_DIR / "seed").glob("*.json")):
        rec = json.loads(p.read_text())
        if rec["bw_gbps"] in wanted:
            store.put_trial(rec)
    for r in store.ok_trials():
        store.update_best(r["bw_gbps"], r["cost"])

    print(f"loaded {len(store.all_trials())} seed trials "
          f"(initial_offsets only: {sorted(wanted)})")
    print(f"best so far: {store.best()}")

    agent = Agent(LLM.from_file(KEY_JSON_PATH), store=store, space=space)

    diagnosis = agent.diagnose()
    print("\n === Diagnose ===")
    print(json.dumps(diagnosis, indent=2, ensure_ascii=False))

    print("\n === next_proposal ===")
    choice = agent.propose(diagnosis=diagnosis, rnd=1)
    print(json.dumps(choice, indent=2, ensure_ascii=False))

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "step6_decision.json").write_text(
        json.dumps({"diagnosis": diagnosis, "proposal": choice}, indent=2, ensure_ascii=False)
    )

    if choice.get("stop"):
        print("\nagent stopped early")
    else:
        assert choice["bw_gbps"] in agent._untried(), "proposal must be untried"
        print(f"\nchosen bw = {choice['bw_gbps']} GB/s (untried)")
    print("STEP6_OK")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

