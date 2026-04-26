# Two-Hour Firmness Experiment Protocol

## Objective
Run a short but strong experiment suite that completes in about two hours while preserving:
- seed coverage
- per-round traceability
- communication accounting
- failure visibility

## Profile
Use mode firm2h in scripts/run_full_experiments.py.

Profile values:
- methods: flexfl, fedavg, moon, scaffold
- datasets: cifar10, cifar100
- alphas: 0.1, 0.5
- seeds: 42, 43, 44
- rounds: 6
- local_epochs: 2
- max_samples_per_client: 256

Total runs: 48

## Commands
Windows PowerShell:

1) Start execution

.\.venv\Scripts\python.exe scripts\run_full_experiments.py --mode firm2h --phase full --max-workers 2

2) Live progress monitor (continuous)

.\.venv\Scripts\python.exe scripts\monitor_run_full_progress.py --watch --interval 5

3) If interrupted, resume without rerunning completed hashes

.\.venv\Scripts\python.exe scripts\run_full_experiments.py --mode firm2h --phase full --max-workers 2

4) Requeue only previously failed hashes

.\.venv\Scripts\python.exe scripts\run_full_experiments.py --mode firm2h --phase full --max-workers 2 --rerun-failed

## Artifacts to review
- artifacts/stats/current_plan.json
- artifacts/stats/live_progress.json
- artifacts/stats/execution_summary.json
- artifacts/stats/statistics.json
- artifacts/traceability/traceability_table.json
- artifacts/final_claim.json
- artifacts/success.json
- artifacts/failed.json
- artifacts/runs/*.json

## Acceptance checks
- success + failed equals requested_runs
- every SUCCESS run has rounds_completed == rounds_expected
- every SUCCESS run has communication.total_bytes > 0
- failed runs include explicit error text

## Notes
- This protocol does not change model/training logic.
- It changes execution profile and monitoring/logging only.
