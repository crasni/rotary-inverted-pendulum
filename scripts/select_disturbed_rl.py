"""Freeze safety-first checkpoint choices using validation reports only."""
import argparse
import json
from pathlib import Path


def rank(result):
    cases = result["cases"].values()
    return (-sum(c["safety_stops"] for c in cases),
            sum(c["passed"] for c in cases),
            sum(c["mean_longest_balance_s"] for c in cases) / len(result["cases"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = {}
    signature = None
    for path in args.reports:
        report = json.loads(path.read_text())
        current = [report[key] for key in ("seed", "episodes", "scenarios", "input_sha256")]
        if signature is not None and current != signature:
            parser.error("Validation reports must use identical seeds, episodes, scenarios and inputs")
        signature = current
        for label, result in report["results"].items():
            if label in candidates or set(result["cases"]) != set(report["scenarios"]):
                parser.error("Require unique candidate labels and complete scenario results")
            candidates[label] = result
    selected = {}
    for algorithm in sorted({r["algorithm"] for r in candidates.values()}):
        labels = [k for k, r in candidates.items() if r["algorithm"] == algorithm]
        best = max(labels, key=lambda label: rank(candidates[label]))
        selected[algorithm] = {"label": best, **candidates[best], "rank": rank(candidates[best])}
    with args.output.open("x") as stream:
        json.dump({"rule": "Fewest safety stops, most passes, longest mean balance; stable ties",
                   "reports": [str(p) for p in args.reports], "selected": selected,
                   "ranking": {k: rank(r) for k, r in candidates.items()}}, stream, indent=2)
        stream.write("\n")
    for algorithm, result in selected.items():
        print(algorithm, result["label"], result["rank"])


if __name__ == "__main__":
    main()
