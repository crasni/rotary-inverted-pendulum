# Hardware results

Every real-pendulum run kept for the submission lives here, one folder per
algorithm: `docs/results/hardware/<algorithm>/<YYYYMMDD_HHMMSS>/`. The tables
are generated; do not edit them by hand.

## Which filmed run goes in the report (learning methods)

| Algorithm | **Use this run (filmed)** | Result | Older filmed run, superseded |
| --- | --- | --- | --- |
| SAC (reward-only rnA) | [`sac/20261109_004713`](sac/20261109_004713/) | **pass**, 13.12 s | `sac/20261108_211936` (v6b, fail 1.06 s) |
| DQN (dqn_rn05) | [`dqn/20261109_010341`](dqn/20261109_010341/) | **pass**, 13.56 s | `dqn/20261108_212009` (dqn_long, fail 0.00 s) |
| PPO (ppo_ecG) | [`ppo/20261109_013508`](ppo/20261109_013508/) | **pass**, 13.14 s | `ppo/20261108_211833` (ppo_long, fail 0.44 s) |

The other runs in each table are the honest attempt count (the assignment asks
to count every attempted trial), not report videos. The 2026-11-09 passes came
after the loose pendulum was re-secured (see `docs/sim-fit-upright.md`).

## Adding a run

Hardware runs are written to `runs/`, which git ignores. To keep one, including
failures you want to report:

```bash
.venv/bin/python scripts/promote_hardware_run.py runs/<run folder>
.venv/bin/python scripts/promote_hardware_run.py runs/<run folder> --video path/to/video.mp4 --note "filmed"
git add docs/results/hardware && git commit
```

The script copies `run.json`, `summary.json` and `episode_000.npz`, adds the
video (under 100 MB, otherwise put a link in `--note`), and rebuilds these
indexes. Name `--output` folders `<algorithm>_<preset>_$(date +%Y%m%d_%H%M%S)`
as the runbooks do; the script needs the timestamp at the end.

## Status by algorithm

The rubric needs one hardware success per category (classical and
learning-based) and, for the advanced challenge, a success for every algorithm.

<!-- BEGIN GENERATED: scripts/promote_hardware_run.py -->
| Category | Algorithm | Passed / attempts | Best pass | Best-pass video |
| --- | --- | --- | --- | --- |
| Classical | [PID](pid/) | 1 / 8 | [`20261107_161650`](pid/20261107_161650/), 10.42 s | not committed |
| Classical | [LQR](lqr/) | 4 / 18 | [`20261107_030202`](lqr/20261107_030202/), 12.84 s | not committed |
| Classical | [ILQR](ilqr/) | 1 / 10 | [`20261107_161026`](ilqr/20261107_161026/), 13.26 s | not committed |
| Classical | [MPC](mpc/) | 1 / 28 | [`20261107_155613`](mpc/20261107_155613/), 10.66 s | not committed |
| Learning | [PPO](ppo/) | 3 / 17 | [`20261109_012337`](ppo/20261109_012337/), 13.52 s | not committed |
| Learning | [SAC](sac/) | 3 / 13 | [`20261109_004642`](sac/20261109_004642/), 13.18 s | not committed |
| Learning | [DQN](dqn/) | 2 / 4 | [`20261109_010341`](dqn/20261109_010341/), 13.56 s | not committed |
<!-- END GENERATED -->
