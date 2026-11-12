from __future__ import annotations

import json
import yaml

from pathlib import Path
from typing import Optional, Dict, Any, List

import sqlite3
from .path import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS trials (
    fingerprint TEXT PRIMARY KEY,
    bw_gbps     INTEGER NOT NULL,
    channel     INTEGER NOT NULL,
    density_gb  INTEGER NOT NULL,
    status      TEXT NOT NULL,
    cost        REAL,
    latency_s   REAL,
    energy_j    REAL,
    record      TEXT NOT NULL,
    created_at  TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS decisions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    round       INTEGER NOT NULL,
    chosen_bw   INTEGER NOT NULL,
    reason      TEXT,
    evidence    TEXT,
    created_at  TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS best (
    id          INTEGER PRIMARY KEY CHECK (id = 1),
    bw_gbps     INTEGER NOT NULL,
    cost        REAL NOT NULL,
    updated_at  TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

class Store:
    def __init__(self, path:Path=DB_PATH)->None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def put_trial(self, record:Dict[str, Any])->bool:
        """return True if new, False if fingerprint is already exists"""
        fp = f"bw{record['bw_gbps']}"
        try:
            self.conn.execute(
                "INSERT INTO trials (fingerprint,bw_gbps,channel,density_gb,status,cost,"
                "latency_s,energy_j,record) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    fp, record["bw_gbps"], record["channel"], record["density_gb"],
                    record.get("status", "ok"), record.get("cost"),
                    record.get("latency_s"), record.get("energy_j"),
                    json.dumps(record),
                ),
            )
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def all_trials(self)->List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT record FROM trials ORDER BY bw_gbps"
        ).fetchall()
        return [json.loads(r["record"]) for r in rows]

    def ok_trials(self) -> List[Dict[str, Any]]:
        return [r for r in self.all_trials() if r.get("status") == "ok"]

    def has_bw(self, bw: int) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM trials WHERE bw_gbps=?", (bw,)
        ).fetchone()
        return row is not None

    # ---- decisions ----
    def put_decision(self, rnd: int, bw: int, reason: str, evidence: str) -> None:
        self.conn.execute(
            "INSERT INTO decisions (round,chosen_bw,reason,evidence) VALUES (?,?,?,?)",
            (rnd, bw, reason, evidence),
        )
        self.conn.commit()

    def decisions(self) -> List[Dict[str, Any]]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM decisions ORDER BY id").fetchall()]

    # ---- best ----
    def update_best(self, bw: int, cost: float) -> str:
        """Returns 'inserted' | 'updated' | 'kept'."""
        row = self.conn.execute("SELECT bw_gbps,cost FROM best WHERE id=1").fetchone()
        if row is None:
            self.conn.execute("INSERT INTO best (id,bw_gbps,cost) VALUES (1,?,?)", (bw, cost))
            self.conn.commit()
            return "inserted"
        if cost < row["cost"]:
            self.conn.execute("UPDATE best SET bw_gbps=?,cost=? WHERE id=1", (bw, cost))
            self.conn.commit()
            return "updated"
        return "kept"

    def best(self) -> Optional[Dict[str, Any]]:
        row = self.conn.execute("SELECT * FROM best WHERE id=1").fetchone()
        return dict(row) if row else None