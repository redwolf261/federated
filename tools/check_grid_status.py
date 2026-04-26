import os, glob, json
from datetime import datetime

run_dir = 'outputs/locked_cifar10_grid/runs'
files = sorted(glob.glob(os.path.join(run_dir, '*.json')))
print(f"Total completed runs: {len(files)}")
print("-" * 60)

for f in files:
    name = os.path.basename(f)
    mtime = datetime.fromtimestamp(os.path.getmtime(f)).strftime('%Y-%m-%d %H:%M:%S')
    try:
        with open(f) as fp:
            data = json.load(fp)
        rounds = len(data.get('per_round', []))
        print(f"{name:30s} | rounds={rounds:2d} | {mtime}")
    except Exception as e:
        print(f"{name:30s} | ERROR: {e}")
