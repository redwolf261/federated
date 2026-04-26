import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))
from scripts.phase2_q1_validation import run_moon
import time

print('Testing MOON with local_epochs=1...')
start = time.time()
r = run_moon(
    dataset_name='cifar10',
    num_classes=10,
    num_clients=10,
    rounds=2,
    local_epochs=1,
    seed=42,
    alpha=0.1,
    lr=0.003,
    batch_size=64,
    max_samples=20000,
    mu=1.0,
    temperature=0.5,
    return_trace=True
)
elapsed = time.time() - start
print(f'MOON test completed in {elapsed:.1f}s')
print(f'Keys: {list(r.keys())}')
print(f'Final accuracy: {r.get("mean_accuracy", 0):.4f}')

