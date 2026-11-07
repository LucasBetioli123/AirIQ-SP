"""Log of every agent run (JSONL): question, retrieved passages, tool calls, answer,
latency, tokens and estimated cost. Keys stay in Portuguese (data schema)."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from src.common.config import data_path


def log_run(run: dict, path: str | None = None) -> None:
    p = path or os.environ.get("AGENT_LOG") or data_path("logs", "agent_runs.jsonl")
    run = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), **run}
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(run, ensure_ascii=False, default=str) + "\n")
