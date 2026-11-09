"""Copy hardware runs from the ignored runs/ folder into the submission and
rebuild the hardware result indexes.

Every hardware result lives at docs/results/hardware/<algorithm>/<YYYYMMDD_HHMMSS>/.
The indexes (docs/results/hardware/README.md and one README per algorithm)
are generated from the runs' summary.json files, so never edit their tables.

    # keep a run (copies it, then reindexes)
    .venv/bin/python scripts/promote_hardware_run.py runs/mpc_transfer_20261108_101500

    # keep a run and its video, with a note shown in the index
    .venv/bin/python scripts/promote_hardware_run.py runs/<run> --video ~/Videos/run.mp4 --note "filmed"

    # only rebuild the indexes
    .venv/bin/python scripts/promote_hardware_run.py --reindex
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
HARDWARE = ROOT / "docs/results/hardware"
ALGORITHMS = {
    "pid": "Classical", "lqr": "Classical", "ilqr": "Classical", "mpc": "Classical",
    "ppo": "Learning", "sac": "Learning", "dqn": "Learning", "custom": "Optional",
}
BEGIN, END = "<!-- BEGIN GENERATED: scripts/promote_hardware_run.py -->", "<!-- END GENERATED -->"
GITHUB_FILE_LIMIT = 95 * 1024 * 1024
RUN_FILES = ("run.json", "summary.json", "episode_000.npz")


def stamp(run_dir):
    match = re.search(r"(\d{8}_\d{6})$", run_dir.name)
    if not match:
        sys.exit(f"{run_dir}: name must end in YYYYMMDD_HHMMSS (the CLI's --output convention)")
    return match.group(1)


def promote(run_dir, video=None, note=None):
    run_dir = Path(run_dir)
    info = json.loads((run_dir / "run.json").read_text())["arguments"]
    if info.get("command") != "hardware":
        sys.exit(f"{run_dir}: not a hardware run")
    target = HARDWARE / info["algorithm"] / stamp(run_dir)
    target.mkdir(parents=True, exist_ok=True)
    for name in RUN_FILES:
        if (run_dir / name).exists():
            shutil.copy2(run_dir / name, target / name)
    if video:
        video = Path(video)
        if video.stat().st_size > GITHUB_FILE_LIMIT:
            sys.exit(f"{video} is over GitHub's 100 MB limit; upload it elsewhere and pass the link "
                     "with --note instead")
        shutil.copy2(video, target / f"video{video.suffix.lower()}")
    if note:
        (target / "note.md").write_text(note.strip() + "\n")
    print(f"Copied {run_dir} -> {target.relative_to(ROOT)}")


def load(run):
    info = json.loads((run / "run.json").read_text())["arguments"]
    summary = {}
    if (run / "summary.json").exists():
        summary = (json.loads((run / "summary.json").read_text()) or [{}])[0]
    video = next((p.name for p in sorted(run.glob("video.*"))), None)
    note = (run / "note.md").read_text().strip() if (run / "note.md").exists() else ""
    return {"name": run.name, "preset": info.get("config"), "checkpoint": info.get("checkpoint"),
            "duration": info.get("duration"), "summary": summary, "video": video, "note": note}


def result(r):
    s = r["summary"]
    if s.get("success"):
        return "✅ pass"
    if not s.get("samples"):
        return "aborted"
    return "fail"


def stop(r):
    reason = r["summary"].get("stop_reason", "—")
    if "fault" in reason:
        return "firmware fault: " + reason.rsplit(",", 1)[-1].strip(" )")
    return reason.split(":")[0]


def best(runs):
    passed = [r for r in runs if r["summary"].get("success")]
    return max(passed, key=lambda r: r["summary"]["longest_balance_s"]) if passed else None


def fmt(value, digits=2):
    return "—" if value is None else f"{value:.{digits}f}"


def algorithm_table(algorithm, runs):
    top = best(runs)
    lines = [f"**{sum(r['summary'].get('success', False) for r in runs)} of {len(runs)} attempts passed.**"]
    if top:
        lines.append(f"Best: [`{top['name']}`]({top['name']}/), {top['summary']['longest_balance_s']:.2f} s "
                     "continuous balance.")
    # For learned controllers the policy file is what was tested.
    learned = ALGORITHMS.get(algorithm) == "Learning"
    extra = " Checkpoint |" if learned else ""
    lines += ["", "| Run | Result | Longest balance (s) | Max arm (°) | Stop | Preset |" + extra + " Video | Note |",
              "| --- | --- | ---: | ---: | --- | --- |" + (" --- |" if learned else "") + " --- | --- |"]
    for r in runs:
        s = r["summary"]
        mark = result(r) + (" **(best)**" if r is top else "")
        preset = Path(r["preset"]).name if r["preset"] else "—"
        video = f"[video]({r['name']}/{r['video']})" if r["video"] else ""
        checkpoint = ""
        if learned:
            path = (r["checkpoint"] or "—").replace("docs/results/", "")
            checkpoint = f" `{path}` |"
        lines.append(f"| [`{r['name']}`]({r['name']}/) | {mark} | {fmt(s.get('longest_balance_s'))} | "
                     f"{fmt(s.get('max_arm_deg'), 1)} | {stop(r)} | `{preset}` |{checkpoint} {video} | {r['note']} |")
    return "\n".join(lines)


STALE = []


def replace_generated(path, generated, default_header, check=False):
    original = path.read_text() if path.exists() else None
    text = original or default_header + f"\n{BEGIN}\n{END}\n" + FILES_NOTE
    if BEGIN not in text:
        text = text.rstrip() + f"\n\n{BEGIN}\n{END}\n"
    head, rest = text.split(BEGIN, 1)
    tail = rest.split(END, 1)[1]
    updated = f"{head}{BEGIN}\n{generated}\n{END}{tail}"
    if updated != original:
        STALE.append(path)
        if not check:
            path.write_text(updated)


FILES_NOTE = """
## Files in each run

- `run.json`: the exact command, hardware profile and rig-model hash.
- `summary.json`: the graded result.
- `episode_000.npz`: the 50 Hz trace (`samples`) and, for newer runs,
  firmware timing and controller internals (`hardware_timing`).
- `video.*` and `note.md`, when added with `--video` and `--note`.
"""


def reindex(check=False):
    rows = []
    for algorithm, category in ALGORITHMS.items():
        folder = HARDWARE / algorithm
        runs = sorted((load(p) for p in folder.iterdir() if (p / "run.json").exists()),
                      key=lambda r: r["name"]) if folder.exists() else []
        if runs:
            replace_generated(folder / "README.md", algorithm_table(algorithm, runs),
                              f"# {algorithm.upper()} hardware runs\n", check)
            top = best(runs)
            passes = sum(r["summary"].get("success", False) for r in runs)
            best_cell = (f"[`{top['name']}`]({algorithm}/{top['name']}/), "
                         f"{top['summary']['longest_balance_s']:.2f} s" if top else "—")
            video = f"[video]({algorithm}/{top['name']}/{top['video']})" if top and top["video"] else (
                "not committed" if top else "—")
            rows.append(f"| {category} | [{algorithm.upper()}]({algorithm}/) | {passes} / {len(runs)} | "
                        f"{best_cell} | {video} |")
        elif category != "Optional":
            rows.append(f"| {category} | {algorithm.upper()} | none committed yet | — | — |")
    table = "\n".join(["| Category | Algorithm | Passed / attempts | Best pass | Best-pass video |",
                       "| --- | --- | --- | --- | --- |", *rows])
    replace_generated(HARDWARE / "README.md", table, "# Hardware results\n", check)
    if check:
        return [p.relative_to(ROOT) for p in STALE]
    print("Reindexed", HARDWARE.relative_to(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", type=Path, help="run folders under runs/")
    parser.add_argument("--video", type=Path, help="video file for a single run")
    parser.add_argument("--note", help="short note shown in the index for a single run")
    parser.add_argument("--reindex", action="store_true", help="only rebuild the indexes")
    parser.add_argument("--check", action="store_true", help="exit 1 if an index is out of date")
    args = parser.parse_args()
    if args.check:
        stale = reindex(check=True)
        if stale:
            sys.exit("Out of date (run with --reindex): " + ", ".join(map(str, stale)))
        print("Hardware indexes are up to date.")
        return
    if not args.runs and not args.reindex:
        parser.error("give run folders or --reindex")
    if (args.video or args.note) and len(args.runs) != 1:
        parser.error("--video and --note apply to exactly one run")
    for run in args.runs:
        promote(run, args.video, args.note)
    reindex()


if __name__ == "__main__":
    main()
