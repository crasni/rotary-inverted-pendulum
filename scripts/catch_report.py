"""List every balance catch in hardware runs and how long it held.

A "catch" is a tick where the controller switches from swing-up to balance.
For each one this prints the state at the catch, the first balance command,
the peak arm speed in the next 0.2 s, and how long balance lasted. It then
scores simple catch-gate rules against the recorded outcomes.

    .venv/bin/python scripts/catch_report.py runs/lqr_coupling_delay_2026* runs/mpc_*_2026*

Works on runs/ folders and on docs/results/hardware/<algorithm>/<run>/ folders.
"""
import argparse
import json
from pathlib import Path

import numpy as np

GOOD_HOLD_S = 4.0


def catches(run_dir):
    data = np.load(run_dir / "episode_000.npz")
    samples, timing = data["samples"], data["hardware_timing"]
    if len(samples) < 50:
        return []
    columns = [str(c) for c in data["hardware_timing_columns"]]
    t, arm, theta, arm_vel, pen_vel, _ = samples.T
    mode = timing[:, columns.index("balance_mode")]
    first_command = timing[:, columns.index("pre_arm_limit_action")]
    algorithm = json.loads((run_dir / "run.json").read_text())["arguments"]["algorithm"]
    rows = []
    for start in np.where((mode[1:] > 0) & (mode[:-1] == 0))[0] + 1:
        released = np.where(mode[start:] == 0)[0]
        end = start + (released[0] if len(released) else len(mode) - start)
        rows.append({
            "run": f"{algorithm} {run_dir.name[-15:]}", "t": t[start], "hold": t[end - 1] - t[start],
            "to_end": not len(released),
            "theta_deg": np.degrees(np.angle(np.exp(1j * theta[start]))), "pen_vel": pen_vel[start],
            "arm_deg": np.degrees(arm[start]), "arm_vel": arm_vel[start], "command": first_command[start],
            "peak_arm_speed": np.abs(arm_vel[start:min(start + 10, end)]).max(),
        })
    return rows


RULES = {
    "peak arm speed in first 0.2 s > 4.3 (outcome, not a gate)": lambda r: r["peak_arm_speed"] > 4.3,
    "|arm speed| > 2.5 at catch": lambda r: abs(r["arm_vel"]) > 2.5,
    "|first command| > 0.7": lambda r: abs(r["command"]) > 0.7,
    "either of the two above (configs/experiments/mpc_my_rig_strict_catch.json)":
        lambda r: abs(r["arm_vel"]) > 2.5 or abs(r["command"]) > 0.7,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    rows = [row for run in parser.parse_args().runs if (run / "episode_000.npz").exists() for row in catches(run)]
    print(f"{'run':20} {'t':>5} {'hold':>5} end {'θ°':>6} {'θ̇':>6} {'arm°':>6} {'arm v':>6} {'cmd':>6} {'peak v':>6}")
    for r in sorted(rows, key=lambda r: -r["hold"]):
        print(f"{r['run']:20} {r['t']:5.2f} {r['hold']:5.2f} {'*' if r['to_end'] else ' ':3} {r['theta_deg']:6.1f} "
              f"{r['pen_vel']:6.2f} {r['arm_deg']:6.1f} {r['arm_vel']:6.2f} {r['command']:6.2f} {r['peak_arm_speed']:6.2f}")
    good = lambda r: r["hold"] >= GOOD_HOLD_S
    print(f"\n{sum(map(good, rows))} of {len(rows)} catches held at least {GOOD_HOLD_S:.0f} s.")
    for name, rule in RULES.items():
        hit = [r for r in rows if rule(r)]
        rest = [r for r in rows if not rule(r)]
        print(f"{name}: {sum(map(good, hit))}/{len(hit)} held when true, {sum(map(good, rest))}/{len(rest)} when false")


if __name__ == "__main__":
    main()
