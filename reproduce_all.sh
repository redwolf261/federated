#!/usr/bin/env bash
set -euo pipefail

# Reproduce all Q1 blueprint phases from a clean environment.
# Usage:
#   bash reproduce_all.sh

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-./.venv/Scripts/python.exe}"
if [ ! -f "$PYTHON_BIN" ]; then
  echo "Python executable not found at $PYTHON_BIN"
  exit 1
fi

echo "[1/4] Installing dependencies"
"$PYTHON_BIN" -m pip install -r requirements.txt

echo "[2/4] Checking required datasets"
"$PYTHON_BIN" - <<'PY'
from pathlib import Path

required = [
    Path("dataset/femnist/train-00000-of-00001.parquet"),
    Path("dataset/cifar-100-python/train"),
    Path("dataset/cifar-100-python/test"),
    Path("dataset/cifar-100-python/meta"),
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise SystemExit("Missing required dataset files:\n" + "\n".join(missing))
print("Dataset integrity precheck passed")
PY

echo "[3/4] Running all blueprint phases"
"$PYTHON_BIN" scripts/execute_q1_blueprint.py --phase all --rounds 10 --max-samples-per-client 2000 --seeds 0 1 2 3 4

echo "[4/4] Printing final artifact index"
"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

out = Path("outputs/q1_blueprint")
summary_path = out / "execution_summary.json"
trace_path = out / "traceability_table.json"

summary = json.loads(summary_path.read_text(encoding="utf-8"))
print("Created:", summary.get("created_at"))
print("Updated:", summary.get("updated_at"))
print("Phases:", sorted(summary.get("phases", {}).keys()))
print("Summary:", summary_path)
print("Traceability:", trace_path)
PY

echo "Reproduction run complete. Check outputs/q1_blueprint and experiments/."
