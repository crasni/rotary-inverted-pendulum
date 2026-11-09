"""Build a durable inventory of runs/ before pruning large artifacts.

Writes (all committed, so they survive deleting runs/):
  docs/run-history/runs_index.csv     one row per run directory, metadata + metrics
  docs/run-history/runs_index.md      same, grouped by family, human-readable
  docs/run-history/file_manifest.csv  every file under runs/: path, bytes, sha256 (for prunable files)
  docs/run-history/metadata/          verbatim copy of every small JSON/log/CSV/py/md in runs/

Usage: python scripts/inventory_runs.py [--runs runs] [--out docs/run-history]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import statistics
from datetime import datetime
from pathlib import Path

META_NAMES = ("run.json", "batch.json", "session.json", "version.json", "training.json",
              "pretraining.json", "metadata.json", "summary.json")
# Text-ish files copied verbatim into the committed archive (small, irreplaceable).
COPY_SUFFIXES = {".json", ".log", ".csv", ".py", ".md", ".sh", ".txt"}
COPY_MAX_BYTES = 2_000_000
PRUNE_DIR_NAMES = {"checkpoints"}
PRUNE_FILE_SUFFIXES = {".pkl"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # keep going; record the failure
        return {"_unreadable": str(exc)}


REPO_TEXT_SUFFIXES = {".md", ".py", ".json", ".sh", ".toml", ".txt", ".csv", ".yaml", ".yml"}
SKIP_DIRS = {"runs", ".venv", ".git", "__pycache__", "run-history"}


def referenced_paths(repo: Path) -> set[str]:
    """Every 'runs/...' path mentioned in committed text files; such files are never pruned."""
    refs: set[str] = set()
    for p in repo.rglob("*"):
        if p.is_file() and p.suffix in REPO_TEXT_SUFFIXES and not SKIP_DIRS & set(p.relative_to(repo).parts):
            try:
                refs.update(m.rstrip(".,;:)'\"`") for m in re.findall(r"runs/[A-Za-z0-9_./-]+", p.read_text(errors="ignore")))
            except OSError:
                pass
    return {r[len("runs/"):] for r in refs}


KEEP: set[str] = set()


def is_prunable(rel: Path) -> bool:
    if str(rel) in KEEP:
        return False
    return any(p in PRUNE_DIR_NAMES for p in rel.parts) or rel.suffix in PRUNE_FILE_SUFFIXES


def stamp_from_name(name: str) -> str:
    m = re.search(r"(20\d{6})_(\d{6})", name)
    if m:
        return datetime.strptime("".join(m.groups()), "%Y%m%d%H%M%S").strftime("%Y-%m-%d %H:%M")
    m = re.search(r"(20\d{6})", name)
    return datetime.strptime(m.group(1), "%Y%m%d").strftime("%Y-%m-%d") if m else ""


def family(rel: Path) -> str:
    top = rel.parts[0]
    if top == "curriculum":
        if len(rel.parts) > 1 and rel.parts[1] == "versions":
            return "curriculum/versions"
        return "curriculum/batches"
    return re.sub(r"_?(20\d{6}(_\d{6})?|\d{2,3})$", "", top) or top


def flat_args(meta: dict) -> dict:
    out = {}
    for name in ("run.json",):
        args = meta.get(name, {}).get("arguments", {}) if isinstance(meta.get(name), dict) else {}
        out.update({k: v for k, v in args.items() if k in ("command", "algorithm", "config", "checkpoint", "params", "duration")})
    for name in ("batch.json", "session.json"):
        d = meta.get(name)
        if isinstance(d, dict):
            for k in ("checkpoint", "stage", "deterministic", "episodes_requested", "episodes_completed",
                      "stop_reason", "learn", "episodes", "learning_rate", "started"):
                if k in d:
                    out[k] = d[k]
    for name in ("version.json", "training.json", "pretraining.json"):
        d = meta.get(name)
        if isinstance(d, dict):
            for k in ("method", "parent", "source", "steps", "seed", "real_fraction", "learning_rate"):
                if k in d:
                    out[k] = d[k]
            if "batches" in d:
                out["n_batches"] = len(d["batches"])
    return out


def metrics(summary) -> dict:
    eps = summary if isinstance(summary, list) else [summary] if isinstance(summary, dict) else []
    eps = [e for e in eps if isinstance(e, dict)]
    if not eps:
        return {}
    bf = [e["balanced_fraction"] for e in eps if isinstance(e.get("balanced_fraction"), (int, float))]
    lb = [e["longest_balance_s"] for e in eps if isinstance(e.get("longest_balance_s"), (int, float))]
    return {
        "episodes": len(eps),
        "successes": sum(bool(e.get("success")) for e in eps),
        "mean_balanced_frac": round(statistics.mean(bf), 3) if bf else "",
        "best_longest_balance_s": round(max(lb), 2) if lb else "",
        "stop_reasons": ";".join(sorted({str(e.get("stop_reason")) for e in eps if "stop_reason" in e})),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="docs/run-history")
    a = ap.parse_args()
    runs, out = Path(a.runs), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta_out = out / "metadata"
    if meta_out.exists():
        shutil.rmtree(meta_out)

    KEEP.update(referenced_paths(Path(".")))
    # A training dir with checkpoints but no final policy.zip keeps its newest checkpoint.
    for ck in runs.rglob("checkpoints"):
        zips = sorted(ck.glob("*.zip"), key=lambda z: [int(n) for n in re.findall(r"\d+", z.name)] or [0])
        if zips and not (ck.parent / "policy.zip").exists():
            KEEP.add(str(zips[-1].relative_to(runs)))
    files = sorted(p for p in runs.rglob("*") if p.is_file())
    kept_refs = sorted(str(p.relative_to(runs)) for p in files if str(p.relative_to(runs)) in KEEP
                       and (any(x in PRUNE_DIR_NAMES for x in p.relative_to(runs).parts) or p.suffix in PRUNE_FILE_SUFFIXES))
    (out / "kept_referenced.txt").write_text("\n".join(kept_refs) + "\n")

    # 1. Manifest of every file; hash the ones we plan to delete.
    total = prunable = 0
    with (out / "file_manifest.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "bytes", "mtime", "prunable", "sha256"])
        for p in files:
            rel = p.relative_to(runs)
            st = p.stat()
            pr = is_prunable(rel)
            total += st.st_size
            prunable += st.st_size if pr else 0
            w.writerow([str(rel), st.st_size, datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                        int(pr), sha256(p) if (pr or p.suffix in {".zip", ".pt"}) else ""])

    # 2. Verbatim copy of small text artifacts (never prunable ones).
    copied = 0
    for p in files:
        rel = p.relative_to(runs)
        if is_prunable(rel) or p.suffix not in COPY_SUFFIXES or p.stat().st_size > COPY_MAX_BYTES:
            continue
        dst = meta_out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
        copied += 1

    # 3. One row per run directory (any dir directly holding a metadata file, or a policy).
    run_dirs = sorted({p.parent for p in files
                       if (p.name in META_NAMES or p.name in ("policy.zip", "actor_weights.pt"))
                       and not is_prunable(p.relative_to(runs))})
    rows = []
    for d in run_dirs:
        rel = d.relative_to(runs)
        meta = {n: load(d / n) for n in META_NAMES if (d / n).exists()}
        dfiles = [p for p in d.rglob("*") if p.is_file()]
        size = sum(p.stat().st_size for p in dfiles)
        pr = sum(p.stat().st_size for p in dfiles if is_prunable(p.relative_to(runs)))
        mt = max((p.stat().st_mtime for p in dfiles), default=0)
        row = {
            "run": str(rel),
            "family": family(rel),
            "when": stamp_from_name(str(rel)) or datetime.fromtimestamp(mt).strftime("%Y-%m-%d %H:%M") + " (mtime)",
            "meta_files": " ".join(sorted(meta)),
            "has_policy": int((d / "policy.zip").exists()),
            "policy_sha256": sha256(d / "policy.zip") if (d / "policy.zip").exists() else "",
            "n_files": len(dfiles),
            "mb_total": round(size / 1e6, 2),
            "mb_pruned": round(pr / 1e6, 2),
        }
        row.update(metrics(meta.get("summary.json")))
        row.update({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in flat_args(meta).items()})
        rows.append(row)

    cols = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    with (out / "runs_index.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # 4. Markdown view grouped by family.
    show = ["run", "when", "algorithm", "config", "checkpoint", "parent", "source", "method", "steps", "stage", "episodes", "successes",
            "mean_balanced_frac", "best_longest_balance_s", "has_policy", "mb_pruned"]
    lines = ["# runs/ inventory", "",
             f"Generated {datetime.now():%Y-%m-%d %H:%M} by `scripts/inventory_runs.py` before pruning.",
             f"{len(files)} files, {total/1e9:.2f} GB total; {prunable/1e9:.2f} GB marked prunable "
             f"(`checkpoints/` dirs and `*.pkl`). {len(rows)} run directories. "
             f"{copied} small text files copied verbatim to `metadata/`.",
             "", "Full columns (method, seeds, learning rates, policy hashes…) are in `runs_index.csv`; "
             "every file that existed, with sha256 of deleted ones, is in `file_manifest.csv`.", ""]
    for fam in sorted({r["family"] for r in rows}):
        fr = sorted((r for r in rows if r["family"] == fam), key=lambda r: (r["when"], r["run"]))
        used = [c for c in show if any(str(r.get(c, "")) not in ("", "0", "0.0") for r in fr) or c == "run"]
        lines += [f"## {fam} ({len(fr)})", "", "| " + " | ".join(used) + " |", "|" + "---|" * len(used)]
        for r in fr:
            cells = [(str(r.get(c, ""))[:110] + ("…" if len(str(r.get(c, ""))) > 110 else "")).replace("|", "/").replace("runs/", "").replace("docs/results/", "results/")
                     for c in used]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    (out / "runs_index.md").write_text("\n".join(lines))

    # Loose top-level files (e.g. tarballs, npz) not inside any run dir.
    print(f"files={len(files)} total_gb={total/1e9:.2f} prunable_gb={prunable/1e9:.2f} "
          f"run_dirs={len(rows)} copied={copied}")


if __name__ == "__main__":
    main()
