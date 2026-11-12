from __future__ import annotations
import json
from typing import Any, Dict, List, Optional

from . import tools
from .llm import LLM
from .prompts import (DIAGNOSE_USER, PROPOSE_USER, SYSTEM_DIAGNOSE, SYSTEM_PROPOSE)

from .store import Store

class Agent:
    def __init__(self, llm:LLM, store:Store, space:Dict[str, Any])->None:
        self.llm = llm
        self.store = store
        self.space = space
        self.legal = sorted(p["channel"]*64 for p in space["points"])

    def _untried(self)->List[int]:
        return [bw for bw in self.legal if not self.store.has_bw(bw)]

    def _space_brief(self)->str:
        return (
            f"One tunable parameter: DRAM bandwidth, {len(self.legal)} discrete values "
            f"from {self.legal[0]} to {self.legal[-1]} GB/s.\n"
            f"All other hardware fields are fixed: {self.space.get('fixed')}\n"
            f"Legal set: {self.legal}"
        )

    def _history_brief(self)->str:
        rows = sorted(self.store.ok_trials(), key=lambda r: r["bw_gbps"])
        if not rows:
            return "(empty)"
        return "\n".join(
            f"  bw={r['bw_gbps']:>5}  cost={r['cost']:.4f}  latency={r['latency_s']:.6e}"
            f"  energy={r['energy_j']:.6e}  dram_util={r.get('dram_util')}"
            f"  dram_stall={r.get('dram_stall_cycles')}"
            for r in rows
        )

     # ---------------- 第一段：诊断 ----------------
    def diagnose(self) -> Dict[str, Any]:
        trend = tools.tool_trend(self.store)
        user = DIAGNOSE_USER.format(
            space=self._space_brief(),
            trials=self._history_brief(),
            trend=json.dumps(trend["steps"], indent=2),
            best=json.dumps(self.store.best(), indent=2),
        )
        result = self.llm.chat_json(SYSTEM_DIAGNOSE, user)
        if not result.get("evidence"):
            raise ValueError(f"diagnosis has no evidence: {result}")
        return result

    # ---------------- 第二段：提案 ----------------
    def propose(self, diagnosis: Dict[str, Any], rnd: int) -> Dict[str, Any]:
        untried = self._untried()
        if not untried:
            return {"stop": True, "reason": "every legal bandwidth value has been simulated"}

        user = PROPOSE_USER.format(
            diagnosis=json.dumps(diagnosis, indent=2),
            history=self._history_brief(),
            untried=untried,
            best=json.dumps(self.store.best(), indent=2),
        )

        last_error: Optional[str] = None
        for attempt in range(3):
            if last_error:
                user_with_error = user + f"\n\n## Your previous answer was rejected\n{last_error}\n"
            else:
                user_with_error = user

            result = self.llm.chat_json(SYSTEM_PROPOSE, user_with_error)

            if result.get("stop") is True:
                return {"stop": True, "reason": result.get("reason", "agent chose to stop")}

            bw = result.get("bw_gbps")
            try:
                bw = int(bw)
            except (TypeError, ValueError):
                last_error = f"bw_gbps must be an integer, got {bw!r}"
                continue

            check = tools.tool_validate(self.store, bw, self.space["points"])
            if not check["ok"]:
                last_error = "; ".join(check["errors"])
                continue

            self.store.put_decision(
                rnd, bw,
                reason=str(result.get("reason", "")),
                evidence=json.dumps(diagnosis.get("evidence", [])),
            )
            return {"stop": False, "bw_gbps": bw, **result}

        raise ValueError(f"agent failed to produce a legal proposal in 3 attempts: {last_error}")