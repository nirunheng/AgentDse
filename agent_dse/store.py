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
    chosen_fp   TEXT_NOT_NULL,
    chosen_pt   TEXT_NOT_NULL,
    reason      TEXT,
    evidence    TEXT,
    created_at  TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS best (
    id          INTEGER PRIMARY KEY CHECK (id = 1),
    fingerprint TEXT NOT NULL,
    point       TEXT NOT NULL,
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
        fp = record["fingerprint"]
        try:
            self.conn.execute(
                "INSERT INTO trials (fingerprint,status,cost,latency_s,energy_j,record)"
                "VALUES (?,?,?,?,?,?)",
                (
                    fp,
                    record.get("status", "ok"),
                    record.get("cost"),
                    record.get("latency_s"),
                    record.get("energy_j"),
                    json.dumps(record),
                ),
            )
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def all_trials(self)->List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT record FROM trials ORDER BY created_at"
        ).fetchall()
        return [json.loads(r["record"]) for r in rows]

    def ok_trials(self) -> List[Dict[str, Any]]:
        return [r for r in self.all_trials() if r.get("status") == "ok"]

    def has_point(self, fingerprint:str)->bool:
        row = self.conn.execute(
            "SELECT 1 FROM trials WHERE fingerprint=?", (fingerprint,)
        ).fetchone()
        return row is not None
        

    # ---- decisions ----
    def put_decision(self, rnd: int, point: Dict[str, Any], fingerprint: str, reason: str, evidence: str) -> None:
        self.conn.execute(
            "INSERT INTO decisions (round,chosen_fp,chosen_pt,reason,evidence)"
            " VALUES (?,?,?,?,?)",
            (rnd, fingerprint, json.dumps(point), reason, evidence),
        )
        self.conn.commit()

    def decisions(self) -> List[Dict[str, Any]]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM decisions ORDER BY id").fetchall()]

    # ---- best ----
    def update_best(self, point: Dict[str, Any], fingerprint:str, cost: float) -> str:
        """Returns 'inserted' | 'updated' | 'kept'."""
        row = self.conn.execute("SELECT cost FROM best WHERE id=1").fetchone()
        if row is None:
            self.conn.execute(
                "INSERT INTO best (id,fingerprint,point,cost) VALUES (1,?,?,?)",
                (fingerprint, json.dumps(point), cost)
                )
            self.conn.commit()
            return "inserted"
        
        # if newly best
        if cost < row["cost"]:
            self.conn.execute(
                "UPDATE best SET fingerprint=?, point=?, cost=? WHERE id=1", 
                (fingerprint, json.dumps(point), cost)
                )
            self.conn.commit()
            return "updated"
        return "kept"

    def best(self) -> Optional[Dict[str, Any]]:
        row = self.conn.execute("SELECT * FROM best WHERE id=1").fetchone()
        if row is None:
            return None
        else:
            return{
                "fingerprint":row["fingerprint"],
                "point":json.dumps(row["point"]),
                "cost":row["cost"]
            }