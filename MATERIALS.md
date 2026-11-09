# Where to find things

This index maps Group 8's HW1 materials to the grading items in the homework
PDF (`RPL2026_hw1.pdf`, "Grading"; also in the [README](README.md#what-reportmd-must-cover)).
[Report.md](Report.md) is the graded report. Everything below is committed
unless marked **local only**.

## Successful results so far

| Algorithm | Real pendulum | Simulation |
| --- | --- | --- |
| **LQR** | **Passed 4 of 18 attempts** (3 of 16 on the night of the pass, 1 of 2 on the afternoon of 11-07). Best: `20261107_030202`, 12.84 s, filmed. [Runs](docs/results/hardware/lqr/README.md) · [write-up](Report.md#lqr-hardware-pass-2026-11-07) · [how to repeat](docs/lqr-hardware-runbook.md) · code tag `lqr-hw-pass-20261107` | Passes |
| **MPC** | **Passed.** `20261107_155613`, 10.66 s, with [`mpc_my_rig_transfer.json`](configs/mpc_my_rig_transfer.json) (1 of 12 attempts with that preset, 9 of which ran the full 15 s; 28 MPC attempts in all). [Runs](docs/results/hardware/mpc/README.md) · [write-up](Report.md#mpc-hardware-pass-2026-11-07) · code tag `mpc-hw-pass-20261107` | Passes |
| **iLQR** | **Passed.** `20261107_161026`, 13.26 s, with [`ilqr_my_rig_lqr_match.json`](configs/ilqr_my_rig_lqr_match.json) (1 of 5 attempts with that preset; 10 iLQR attempts in all). [Runs](docs/results/hardware/ilqr/README.md) · [write-up](Report.md#ilqr-hardware-pass-2026-11-07) · code tag `ilqr-hw-pass-20261107` | Passes |
| **PID** | **Passed.** `20261107_161650`, 10.42 s, with [`pid_my_rig_lqr_match.json`](configs/pid_my_rig_lqr_match.json) (1 of 7 attempts with that preset; 8 PID attempts in all). Gains derived from the LQR solution ([why](Report.md#matching-pid-and-ilqr-to-the-lqr-pass)). [Runs](docs/results/hardware/pid/README.md) · [write-up](Report.md#pid-hardware-pass-2026-11-07) · code tag `pid-hw-pass-20261107` | Passes |
| PPO, SAC, DQN | Not yet tried on hardware. Rig-tuned policies from `rl/rig-training`: [SAC](docs/results/rig_rl_disturbance/sac_long/policy.zip), [DQN](docs/results/rig_rl_disturbance/dqn_long/policy.zip), [PPO](docs/results/rig_rl_disturbance/ppo_long/policy.zip), run with [`configs/sac_my_rig.json`](configs/sac_my_rig.json) ([training and selection](docs/rl-disturbance-tuning.md)) | Held-out: SAC 63/80, DQN 55/80, PPO 40/80 |

The live table for every algorithm is [docs/results/hardware/README.md](docs/results/hardware/README.md).

## Grading items

| Item (weight) | What it needs | Where it is | Status |
| --- | --- | --- | --- |
| Assembly (10%) | Rig documentation | [Report.md](Report.md#assembly-pending), [hardware guide](docs/hardware.md), [first-start checklist](docs/first-start-checklist.md) | Photos and write-up pending |
| System identification (15%) | Video of the process, results | Rig model [`configs/my_rig.json`](configs/my_rig.json), raw recordings and validation in [`docs/results/sysid/`](docs/results/sysid/README.md), [SysID guide](docs/system-identification.md) | **Data committed.** Video and Report section pending |
| Swing-up logic (7%) | Described in Report.md | [Algorithm notes](docs/algorithms.md#classical-controllers), [settings reference](docs/controller-settings.md) | Notes written; Report section to write |
| Per algorithm, 7 × 5% | Simulator GIF beside real video, with metrics | Simulator GIFs in [`docs/results/`](docs/results/); real runs in [`docs/results/hardware/<algorithm>/`](docs/results/hardware/README.md) | LQR: metrics committed, video not yet committed. Others: pending |
| One success per category, 2 × 4% | One classical and one learning-based hardware success | Classical: [LQR](docs/results/hardware/lqr/README.md), [MPC](docs/results/hardware/mpc/README.md), [iLQR](docs/results/hardware/ilqr/README.md), [PID](docs/results/hardware/pid/README.md) | **Classical done.** Learning-based pending |
| AI usage and reflection (5%) | Interactions and reflection | [Report.md](Report.md#ai-usage-and-reflection-draft-log) | Draft |
| Advanced: every algorithm (10%) | Hardware success for all seven | [docs/results/hardware/README.md](docs/results/hardware/README.md) | 4 of 7 (all four classical; PPO, SAC, DQN pending) |
| Advanced: live demo (10%) | Two 30 s runs, Monday 11/9 lecture | Use the [LQR runbook](docs/lqr-hardware-runbook.md) with `--duration 30` | Pending |

## Where new results go

Use these locations so the next success is found the same way as this one.

| Result | Location | How |
| --- | --- | --- |
| Hardware run, pass or fail | `docs/results/hardware/<algorithm>/<YYYYMMDD_HHMMSS>/` | `.venv/bin/python scripts/promote_hardware_run.py runs/<run>` ([details](docs/results/hardware/README.md#adding-a-run)) |
| Real-pendulum video | The run's folder as `video.*`, or a link in its note | Same command with `--video` or `--note` |
| Preset that produced a success | `configs/<algorithm>_my_rig_<name>.json` | Keep it unchanged once used on hardware |
| Superseded preset | `configs/experiments/`, with a row in its [README](configs/experiments/README.md) | `git mv` |
| Learned checkpoint used in a result | `docs/results/<algorithm>_<name>_policy.zip` | `git add -f` (git ignores `*.zip`) |
| Simulator GIF | `docs/results/<algorithm>_<name>.gif` | `pendulum-lab sim ... --gif`, then copy |
| Write-up | The matching section of [Report.md](Report.md) | Link to the files above |

Before committing, run the tests (`.venv/bin/python -m pytest -q`). They check
that the hardware indexes are up to date, the LQR passes still replay exactly,
and the classical controllers' shared code is in sync.

## Code and settings

| Path | Contents |
| --- | --- |
| `src/pendulum_lab/algorithms/` | One file per algorithm. `lqr.py`, `mpc.py`, `ilqr.py` and `pid.py` carry an identical swing-up and hand-off block; edit it in `lqr.py`, then run `scripts/sync_handoff.py`. |
| `configs/` | Current presets: `my_rig.json`, the LQR hardware reference `lqr_my_rig_coupling_delay.json`, the `*_my_rig_transfer.json` presets, the RL configs |
| [`docs/controller-settings.md`](docs/controller-settings.md) | Every classical controller setting, and how to move the LQR setup to another algorithm |
| `scripts/` | Reproduction and maintenance scripts |
| `runs/` | **Local only.** Every raw run, including earlier hardware attempts summarized in [configs/experiments/README.md](configs/experiments/README.md) |

## Not in git yet

- The video of LQR run `030202`, and any videos of MPC run `155613`, iLQR run `161026` and PID run `161650`.
- System identification video.
- `runs/lqr_hardware_pass_20261107_025738.tar.gz`, a local bundle of run 025738.
  `runs/lqr_hardware_best_win` is a local shortcut to the 030202 run.

The TAs clone the **main** branch. `param/v1` is merged into `main`; keep
merging new results so `main` stays complete.
