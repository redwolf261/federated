import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

files = {
    "alpha_0.1": ROOT / "outputs" / "short_experiment_alpha_0.1.json",
    "alpha_1.0": ROOT / "outputs" / "short_experiment_alpha_1.0.json",
}


def load_json(path: Path):
    # PowerShell redirection produced UTF-16 LE text.
    txt = path.read_text(encoding="utf-16")
    return json.loads(txt)


def main():
    out = {}
    for alpha_key, path in files.items():
        data = load_json(path)[alpha_key]
        out[alpha_key] = {}
        for method in ["fedavg", "moon", "scaffold", "flex"]:
            m = data[method]
            per_seed_mean = {
                s: float(v["mean_accuracy"]) for s, v in m["per_seed"].items()
            }
            per_seed_worst = {
                s: float(v["worst_accuracy"]) for s, v in m["per_seed"].items()
            }
            # One curve per method: choose seed 42 curve (first stored run)
            curve = [float(x) for x in m["round_curves"][0]]
            out[alpha_key][method] = {
                "mean_accuracy": float(m["mean_accuracy"]),
                "worst_client_accuracy": float(m["worst_accuracy"]),
                "per_seed_mean_accuracy": per_seed_mean,
                "per_seed_worst_client_accuracy": per_seed_worst,
                "convergence_curve_seed_42": curve,
            }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
