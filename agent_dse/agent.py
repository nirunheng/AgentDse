from __future__ import annotations
import json
from typing import Any, Dict, List, Optional

from . import tools
from .llm import LLM
from .prompts import (DIAGNOSE_USER, PROPOSE_USER, SYSTEM_DIAGNOSE, SYSTEM_PROPOSE)
from .dims import Space

from .store import Store

class Agent:
    def __init__(self, llm:LLM, store:Store, space:Space)->None:
        self.llm = llm
        self.store = store
        self.space = space
        # self.legal = sorted(p["channel"]*64 for p in space["points"])

    def _untried(self)->List[int]:
        return [
            p for p in self.space.all_points()
            if not self.store.has_point(self.space.fingerprint(p))
        ]
        
    def _space_brief(self)->str:
        return self.space.describe()

    def _history_brief(self)->str:
        rows = self.store.ok_trials()
        if not rows:
            return "(empty)"

        rows = sorted(rows, key=lambda r : r.get("cost") or 1e9)
        lines = []
        for r in rows:
            head = " " + self.space.point_brief(r)
            lines.append(
                f"{head} cost={r.get('cost'):.4f}"
                f"latency={r.get('latency_s'):.6e} energy={r.get('energy_j'):.6e}"
                f"dram_util={r.get('dram_util')} matrix_util={r.get('matrix_util')}"
                f"dram_srall={r.get('dram_stall_cycle')}"
            )

        return "\n".join(lines)


     # ---------------- 第一段：诊断 ----------------
    def diagnose(self) -> Dict[str, Any]:
        trend = tools.tool_trend(self.store, self.space)
        user = DIAGNOSE_USER.format(
            space=self._space_brief(),
            trials=self._history_brief(),
            trend=json.dumps(trend["per_axis"], indent=2, ensure_ascii=False),
            best=json.dumps(self.store.best(), indent=2, ensure_ascii=False),
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
            space=self.space,
            diagnosis=json.dumps(diagnosis, indent=2, ensure_ascii=False),
            history=self._history_brief(),
            untried_n=min(30, len(untried)),
            untried_total=len(untried),
            untried="\n".join(
                " " + self.space.point_brief(p) for p in untried[:30]
            ),
            best=json.dumps(self.store.best(), indent=2, ensure_ascii=False)
        )

        last_error: Optional[str] = None
        for attempt in range(3):
            user_with_error = user if not last_error else (
                user + f"\n\n## Your previous answer was rejected\n{last_error}\n"
            )
            result = self.llm.chat_json(SYSTEM_PROPOSE, user_with_error)

            if result.get("stop") is True:
                return {"stop": True,
                        "reason": result.get("reason", "agent chose to stop")}

            # 1) 取出 point 并逐个维度转 int
            raw = result.get("point")
            if not isinstance(raw, dict):
                last_error = f"'point' must be a dict of dimension values, got {raw!r}"
                continue
            point: Dict[str, Any] = {}
            bad = None
            for name in self.space.names:
                if name not in raw:
                    bad = f"missing dimension '{name}'"
                    break
                try:
                    point[name] = int(raw[name])
                except (TypeError, ValueError):
                    bad = f"dimension '{name}' must be an int, got {raw[name]!r}"
                    break
            if bad:
                last_error = bad
                continue

            # 2) 合法性 + 去重
            check = tools.tool_validate(self.store, point, self.space)
            if not check["ok"]:
                last_error = "; ".join(check["errors"])
                continue

            # 3) 落库
            fp = self.space.fingerprint(point)
            self.store.put_decision(
                rnd, point, fp,
                reason=str(result.get("reason", "")),
                evidence=json.dumps(diagnosis.get("evidence", []), ensure_ascii=False),
            )
            return {"stop": False, "point": point, "fingerprint": fp, **result}

        raise ValueError(
            f"agent failed to produce a legal proposal in 3 attempts: {last_error}"
        )
        #     if last_error:
        #         user_with_error = user + f"\n\n## Your previous answer was rejected\n{last_error}\n"
        #     else:
        #         user_with_error = user

        #     result = self.llm.chat_json(SYSTEM_PROPOSE, user_with_error)

        #     if result.get("stop") is True:
        #         return {"stop": True, "reason": result.get("reason", "agent chose to stop")}

        #     bw = result.get("bw_gbps")
        #     try:
        #         bw = int(bw)
        #     except (TypeError, ValueError):
        #         last_error = f"bw_gbps must be an integer, got {bw!r}"
        #         continue

        #     check = tools.tool_validate(self.store, bw, self.space["points"])
        #     if not check["ok"]:
        #         last_error = "; ".join(check["errors"])
        #         continue

        #     self.store.put_decision(
        #         rnd, bw,
        #         reason=str(result.get("reason", "")),
        #         evidence=json.dumps(diagnosis.get("evidence", [])),
        #     )
        #     return {"stop": False, "bw_gbps": bw, **result}

        # raise ValueError(f"agent failed to produce a legal proposal in 3 attempts: {last_error}")
        