import json
from pathlib import Path
import datetime

pm = Path('outputs/failure_mode_coverage/block_M_results.json')
dm = json.loads(pm.read_text())
completed = len(dm)
print(f'Done: {completed}/15')

times = []
for r in dm:
    mins = r.get('wall_time_s', 0) / 60
    times.append(mins)
    print(f"  {r['method']} s{r['seed']}: {mins:.1f} mins")

avg_time = sum(times) / len(times) if times else 45
remaining_runs = 15 - completed
total_remaining_mins = remaining_runs * avg_time
eta = datetime.datetime.now() + datetime.timedelta(minutes=total_remaining_mins)

print(f'\nAvg per run: {avg_time:.1f} mins')
print(f'Remaining: {remaining_runs} runs')
print(f'Est. Remaining Time: {total_remaining_mins/60:.1f} hours')
fmt = "%I:%M %p"
print(f'ETA Timestamp: {eta.strftime(fmt)}')
