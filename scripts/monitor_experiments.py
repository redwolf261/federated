#!/usr/bin/env python3
"""Real-time experiment progress monitor for FLEX-Persona comprehensive experiments."""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

def format_duration(seconds):
    """Format duration in human-readable form."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds//60:.0f}m {seconds%60:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m"

def get_experiment_status():
    """Get current experiment status from registry and directories."""
    workspace = Path.cwd()
    experiments_dir = workspace / "experiments"
    registry_path = experiments_dir / "experiment_registry.json"

    if not registry_path.exists():
        return {"error": "No experiment registry found"}

    with open(registry_path) as f:
        registry = json.load(f)

    # Filter for 10-seed experiments (our main suite)
    target_experiments = [
        exp for exp in registry["experiments"]
        if exp.get("num_seeds") == 10 and "10seed" in exp["experiment_id"]
    ]

    # Expected conditions
    expected_conditions = [
        ("fedavg", "high_het"),
        ("prototype", "high_het"),
        ("fedavg", "low_het"),
        ("prototype", "low_het")
    ]

    status = {
        "total_expected": 4,
        "completed": 0,
        "in_progress": 0,
        "pending": 0,
        "conditions": {},
        "current_activity": None,
        "estimated_completion": None
    }

    # Track completion status for each condition
    condition_status = {}

    for method, regime in expected_conditions:
        key = f"{method}_{regime}"
        condition_status[key] = {
            "method": method,
            "regime": regime,
            "status": "pending",
            "experiment_id": None,
            "seeds_completed": 0,
            "start_time": None,
            "end_time": None,
            "duration": None
        }

    # Process completed and in-progress experiments
    for exp in target_experiments:
        method = exp["method"]
        regime = exp["regime"]
        key = f"{method}_{regime}"

        if key in condition_status:
            condition_status[key]["experiment_id"] = exp["experiment_id"]
            condition_status[key]["start_time"] = exp["start_timestamp"]
            condition_status[key]["end_time"] = exp.get("end_timestamp")

            if exp["status"] == "completed":
                condition_status[key]["status"] = "completed"
                condition_status[key]["seeds_completed"] = exp["num_seeds"]
                status["completed"] += 1

                # Calculate duration
                if exp.get("start_timestamp") and exp.get("end_timestamp"):
                    start = datetime.fromisoformat(exp["start_timestamp"])
                    end = datetime.fromisoformat(exp["end_timestamp"])
                    condition_status[key]["duration"] = (end - start).total_seconds()
            else:
                # Check if experiment is actively running
                exp_dir = experiments_dir / exp["experiment_id"]
                if exp_dir.exists():
                    per_seed_file = exp_dir / "per_seed_results.json"
                    if per_seed_file.exists():
                        try:
                            with open(per_seed_file) as f:
                                per_seed_data = json.load(f)
                            condition_status[key]["seeds_completed"] = len(per_seed_data)
                            condition_status[key]["status"] = "in_progress"
                            status["in_progress"] += 1
                            status["current_activity"] = f"{method.upper()} × {regime.replace('_', ' ').title()}"
                        except:
                            condition_status[key]["status"] = "starting"
                            status["in_progress"] += 1

    # Count pending
    status["pending"] = status["total_expected"] - status["completed"] - status["in_progress"]
    status["conditions"] = condition_status

    # Estimate completion time
    completed_durations = [
        cond["duration"] for cond in condition_status.values()
        if cond["duration"] is not None
    ]

    if completed_durations and status["in_progress"] + status["pending"] > 0:
        avg_duration = sum(completed_durations) / len(completed_durations)
        remaining_conditions = status["in_progress"] + status["pending"]

        # Adjust for in-progress completion
        if status["in_progress"] > 0:
            for cond in condition_status.values():
                if cond["status"] == "in_progress" and cond["start_time"]:
                    start = datetime.fromisoformat(cond["start_time"])
                    elapsed = (datetime.now() - start).total_seconds()
                    remaining_for_current = max(0, avg_duration - elapsed)
                    estimated_remaining = remaining_for_current + (remaining_conditions - 1) * avg_duration
                    break
        else:
            estimated_remaining = remaining_conditions * avg_duration

        status["estimated_completion"] = datetime.now() + timedelta(seconds=estimated_remaining)

    return status

def display_status(status, clear_screen=True):
    """Display formatted status information."""
    if clear_screen and os.name != 'nt':
        os.system('clear')
    elif clear_screen and os.name == 'nt':
        os.system('cls')

    print("[TARGET] FLEX-PERSONA COMPREHENSIVE EXPERIMENT MONITOR")
    print("=" * 80)

    if "error" in status:
        print(f"[ERROR] Error: {status['error']}")
        return

    # Overall progress
    completed = status["completed"]
    total = status["total_expected"]
    progress_pct = (completed / total) * 100

    print(f"[PROGRESS] OVERALL: {completed}/{total} conditions completed ({progress_pct:.0f}%)")

    # Progress bar
    bar_length = 50
    filled = int(progress_pct / 100 * bar_length)
    bar = "=" * filled + "-" * (bar_length - filled)
    print(f"[{bar}] {progress_pct:.1f}%")
    print()

    # Current activity
    if status["current_activity"]:
        print(f"[ACTIVE] CURRENTLY RUNNING: {status['current_activity']}")
    elif status["pending"] > 0:
        print("[WAITING] NEXT: Preparing next condition...")
    else:
        print("[SUCCESS] ALL EXPERIMENTS COMPLETED!")
    print()

    # Condition details
    print("[STATUS] EXPERIMENT CONDITIONS:")
    print("+" + "-" * 33 + "+" + "-" * 10 + "+" + "-" * 11 + "+" + "-" * 14 + "+")
    print("| Condition                       | Status   | Seeds     | Duration     |")
    print("+" + "-" * 33 + "+" + "-" * 10 + "+" + "-" * 11 + "+" + "-" * 14 + "+")

    for key, cond in status["conditions"].items():
        method_display = cond["method"].upper()
        regime_display = cond["regime"].replace("_", " ").title()
        condition_name = f"{method_display} x {regime_display}"

        # Status display
        if cond["status"] == "completed":
            status_icon = "[DONE]"
        elif cond["status"] == "in_progress":
            status_icon = "[RUN]"
        elif cond["status"] == "starting":
            status_icon = "[START]"
        else:
            status_icon = "[WAIT]"

        # Seeds display
        seeds_display = f"{cond['seeds_completed']}/10"

        # Duration display
        if cond["duration"]:
            duration_display = format_duration(cond["duration"])
        elif cond["status"] == "in_progress" and cond["start_time"]:
            start = datetime.fromisoformat(cond["start_time"])
            elapsed = (datetime.now() - start).total_seconds()
            duration_display = format_duration(elapsed) + "*"
        else:
            duration_display = "---"

        print(f"| {condition_name:<31} | {status_icon:<8} | {seeds_display:>9} | {duration_display:>12} |")

    print("+" + "-" * 33 + "+" + "-" * 10 + "+" + "-" * 11 + "+" + "-" * 14 + "+")
    print()

    # Time estimates
    if status["estimated_completion"]:
        est_time = status["estimated_completion"]
        remaining = (est_time - datetime.now()).total_seconds()
        print(f"[TIME] ESTIMATED COMPLETION: {est_time.strftime('%H:%M:%S')} ({format_duration(remaining)} remaining)")

    print(f"[UPDATE] LAST UPDATE: {datetime.now().strftime('%H:%M:%S')}")
    print()
    print("[TIP] Press Ctrl+C to stop monitoring")

def main():
    """Main monitoring function."""
    import argparse

    parser = argparse.ArgumentParser(description="Monitor FLEX-Persona experiment progress")
    parser.add_argument("--watch", "-w", action="store_true", help="Continuous monitoring (updates every 30 seconds)")
    parser.add_argument("--interval", "-i", type=int, default=30, help="Update interval in seconds (default: 30)")
    parser.add_argument("--no-clear", action="store_true", help="Don't clear screen between updates")

    args = parser.parse_args()

    try:
        if args.watch:
            print("Starting continuous monitoring... (Press Ctrl+C to stop)")
            while True:
                status = get_experiment_status()
                display_status(status, clear_screen=not args.no_clear)

                # Exit if all experiments are completed
                if status.get("completed", 0) >= status.get("total_expected", 4):
                    print("[SUCCESS] ALL EXPERIMENTS COMPLETED!")
                    break

                time.sleep(args.interval)
        else:
            status = get_experiment_status()
            display_status(status, clear_screen=False)

    except KeyboardInterrupt:
        print("\n\n[STOP] Monitoring stopped by user")

if __name__ == "__main__":
    main()