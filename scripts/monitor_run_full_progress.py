#!/usr/bin/env python3
"""Live monitor for run_full_experiments artifact progress."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

WORKSPACE = Path(__file__).resolve().parents[1]
STATS_DIR = WORKSPACE / "artifacts" / "stats"
LIVE_PATH = STATS_DIR / "live_progress.json"
PLAN_PATH = STATS_DIR / "current_plan.json"
SUMMARY_PATH = STATS_DIR / "execution_summary.json"


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _pct(done: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return 100.0 * float(done) / float(total)


def _pick(*sources: dict[str, Any], key: str, default: Any) -> Any:
    for src in sources:
        if key in src and src.get(key) is not None:
            return src.get(key)
    return default


def _render_once(no_clear: bool) -> bool:
    plan = _load_json(PLAN_PATH) or {}
    live = _load_json(LIVE_PATH) or {}
    summary = _load_json(SUMMARY_PATH) or {}

    if not no_clear:
        print("\x1bc", end="")

    phase = str(_pick(live, plan, summary, key="phase", default="unknown"))
    mode = str(_pick(live, plan, summary, key="mode", default="unknown"))

    requested = int(_pick(live, plan, summary, key="requested_runs", default=0))
    done = int(_pick(live, summary, key="done_runs", default=_pick(summary, key="executed_runs", default=0)))
    success = int(_pick(live, summary, key="success_runs", default=0))
    failed = int(_pick(live, summary, key="failed_runs", default=0))
    pending = int(_pick(live, key="pending_runs", default=max(requested - done, 0)))
    workers = int(_pick(live, plan, summary, key="max_workers", default=0))
    active = bool(live.get("active", False))

    ts = str(_pick(live, summary, key="updated_at", default=_pick(summary, key="created_at", default="n/a")))
    pct = _pct(done, requested)
    bar_slots = 40
    fill = int((pct / 100.0) * bar_slots)
    bar = "#" * fill + "-" * (bar_slots - fill)

    print("RUN FULL EXPERIMENTS MONITOR")
    print("=" * 80)
    print(f"timestamp: {datetime.now().isoformat()}")
    print(f"phase: {phase} | mode: {mode} | workers: {workers}")
    print(f"progress: [{bar}] {pct:.1f}%")
    print(f"requested={requested} done={done} pending={pending} success={success} failed={failed}")
    print(f"last_artifact_update={ts}")
    print(f"active={active}")

    if not plan and not live and not summary:
        print("\nNo run artifacts found yet. Start run_full_experiments first.")

    return active


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor run_full_experiments progress")
    parser.add_argument("--watch", action="store_true", help="Continuously refresh monitor")
    parser.add_argument("--interval", type=int, default=5, help="Refresh interval seconds")
    parser.add_argument("--no-clear", action="store_true", help="Do not clear terminal between refreshes")
    args = parser.parse_args()

    if not args.watch:
        _render_once(no_clear=True)
        return

    while True:
        is_active = _render_once(no_clear=args.no_clear)
        if not is_active and SUMMARY_PATH.exists():
            break
        time.sleep(max(int(args.interval), 1))


if __name__ == "__main__":
    main()
