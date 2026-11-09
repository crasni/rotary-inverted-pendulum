"""Copy the shared hand-off block from lqr.py into mpc.py, ilqr.py and pid.py.

The assignment keeps each algorithm in its own file, so the four classical
controllers each carry an identical copy of the swing-up, capture, estimation
and arm-limit code between the BEGIN/END SHARED HAND-OFF markers. lqr.py holds
the copy you edit.

    .venv/bin/python scripts/sync_handoff.py          # update the copies
    .venv/bin/python scripts/sync_handoff.py --check  # exit 1 if they differ
"""
import argparse
from pathlib import Path
import sys

ALGORITHMS = Path(__file__).resolve().parents[1] / "src/pendulum_lab/algorithms"
SOURCE = "lqr.py"
COPIES = ("mpc.py", "ilqr.py", "pid.py")
BEGIN = "# ===== BEGIN SHARED HAND-OFF ====="
END = "# ===== END SHARED HAND-OFF ====="


def split(text, name):
    try:
        start = text.index(BEGIN)
        stop = text.index(END, start) + len(END)
    except ValueError:
        raise SystemExit(f"{name}: missing shared hand-off markers") from None
    return text[:start], text[start:stop], text[stop:]


def block(name):
    return split((ALGORITHMS / name).read_text(), name)[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="only report differences")
    args = parser.parse_args()
    canonical = block(SOURCE)
    stale = []
    for name in COPIES:
        path = ALGORITHMS / name
        before, current, after = split(path.read_text(), name)
        if current == canonical:
            continue
        stale.append(name)
        if not args.check:
            path.write_text(before + canonical + after)
    if args.check and stale:
        print("Out of sync with lqr.py:", ", ".join(stale))
        sys.exit(1)
    print("Updated: " + ", ".join(stale) if stale else "All copies match lqr.py.")


if __name__ == "__main__":
    main()
