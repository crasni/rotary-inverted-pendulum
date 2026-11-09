# Learning-based controllers: hardware passes (2026-11-09)

All three learning methods (PPO, SAC, DQN) passed the full task on the real pendulum
through the official runner (`pendulum-lab hardware`, 15 s, from hanging, deterministic,
the policy alone does swing-up and balance, standard arm limiter). Every attempt, pass or
fail, is in the generated indexes under `docs/results/hardware/<algorithm>/`.

## Report runs

| Algorithm | Report run | Result | Checkpoint (committed) | How it was trained |
|---|---|---|---|---|
| SAC | [`sac/20261109_004713`](results/hardware/sac/20261109_004713/) (filmed) | pass, 13.12 s | [`rig_rl_disturbance/sac_rnA_reward_only`](results/rig_rl_disturbance/sac_rnA_reward_only/LINEAGE.md) | reward-only SAC, simulator with rig-like heavy-tailed reading noise |
| DQN | [`dqn/20261109_010341`](results/hardware/dqn/20261109_010341/) (filmed) | pass, 13.56 s | [`rig_rl_disturbance/dqn_rn05`](results/rig_rl_disturbance/dqn_rn05/LINEAGE.md) | DQN, simulator with half-strength rig-like reading noise |
| PPO | [`ppo/20261109_013508`](results/hardware/ppo/20261109_013508/) | pass, 13.14 s | [`rig_rl_disturbance/ppo_ecG`](results/rig_rl_disturbance/ppo_ecG/LINEAGE.md) | PPO, energy-cap shaping + spin-recovery starts + rig-informed randomization |

Videos are kept by the team, not committed. Check which runs were filmed before linking.

## Attempt counts (repaired rig, 2026-11-09, official runner)

| Algorithm | Version | Attempts | Passes | Notes |
|---|---|---|---|---|
| SAC | rnA | 4 | 2 | 1 encoder fault before start, 1 fail at 12.78 s |
| DQN | dqn_rn05 | 2 | 2 | |
| PPO | ppo_long / B / G (and C, F, rn, rn05) | 15 | 3 | ppo_long 1/3, ppo_ecB 1/4 (one of the 4 was stopped by hand, not valid), ppo_ecG 1/2; several runs stopped by hand while spinning; 1 did not start (USB) |

Whole-history totals are in `docs/results/hardware/README.md` (generated).

## What changed on the rig: the pendulum had come loose

Around midnight on 2026-11-08/09 every controller, LQR included, stopped working (about 20 LQR
runs, best 4.5 s, balance tilt jumping −0.7° to +2.9° from run to run). The pendulum
had come loose relative to its shaft and magnet. After it was re-secured (about 00:42),
LQR held 7.7 s with −0.4° tilt again. The deterministic swing-up's second peak had crept from about 59° (Nov 7) to about 68°,
so the loosening may have been developing for a while. **RL results from before the fix
(including earlier PPO/DQN/SAC failures) were probably affected.** The previous PPO
(`ppo_long`) failed on Nov 7–8 and passed on the repaired rig with the same weights.
Details: [`docs/sim-fit-upright.md`](sim-fit-upright.md).

## How each passing policy was trained (rules-relevant facts)

- **Each method is trained only by its own learner** (SB3 SAC / DQN / PPO). No imitation, no
  other algorithm's weights, no other controller's commands.
- **SAC rnA** ([lineage](results/rig_rl_disturbance/sac_rnA_reward_only/LINEAGE.md)):
  SAC-only ancestors. Some ancestors (v2m2–v6b) trained partly on SAC's *own* logged rig
  transitions. Some of those rig episodes began from an LQR *placement* (a reset tool; LQR ticks
  were never training data). v6b and rnA add a **swing-up anchor** loss that keeps the swing-up
  near the parent SAC's own actions. rnA = v6b + 200k SAC steps with rig-like reading noise.
- **DQN dqn_rn05** ([lineage](results/rig_rl_disturbance/dqn_rn05/LINEAGE.md)): simulator only;
  `dqn_long` + 400k DQN steps with half-strength reading noise, random delay, disturbances.
- **PPO ppo_ecG** ([lineage](results/rig_rl_disturbance/ppo_ecG/LINEAGE.md)): simulator only
  (PPO is on-policy). Fixes a **reward exploit**: under the lab reward, spinning earns more
  than hanging, so after a missed catch PPO kept spinning. Training adds an energy-cap
  penalty (reward shaping), spin-recovery start states, random delay / arm lag / coupling, and
  30% of episodes start from real rig *states* (not transitions).
- Reward changes affect training only. The evaluation metric, controller interface and
  hardware safeguards are unchanged.

## Side-by-side material for §7 (simulator GIF + real run, same policy)

The older RL GIFs (`docs/results/{ppo,sac,dqn}_my_rig.gif`) show earlier policies
(ppo_long, v6b, dqn_long). For the passing policies use:

| Algorithm | Simulator GIF | Sim: balance / final-10 s max \|θ\|, \|θ̇\| | Rig run | Rig: balance / final-10 s max \|θ\|, \|θ̇\| |
|---|---|---|---|---|
| SAC | [`sac_sac_rnA_reward_only_my_rig.gif`](results/sac_sac_rnA_reward_only_my_rig.gif) | 12.84 s / 0.32°, 0.10 rad/s | `sac/20261109_004713` | 13.12 s / 3.08°, 2.21 rad/s |
| DQN | [`dqn_dqn_rn05_my_rig.gif`](results/dqn_dqn_rn05_my_rig.gif) | 12.68 s / 0.93°, 0.47 rad/s | `dqn/20261109_010341` | 13.56 s / 4.22°, 3.00 rad/s |
| PPO | [`ppo_ppo_ecG_my_rig.gif`](results/ppo_ppo_ecG_my_rig.gif) | 13.20 s / 5.56°, 1.53 rad/s | `ppo/20261109_013508` | 13.14 s / 7.12°, 2.85 rad/s |

All metrics (first time within 15°, arm excursion, command RMS, …) and the exact simulator
commands are in [`docs/results/report_rl_passes.json`](results/report_rl_passes.json), computed
the same way as `docs/results/report_sim_my_rig.json`. Filmed: SAC 004713 and DQN 010341;
confirm the PPO video with the team.

## Robustness observations (not graded attempts)

- SAC rnA, poked 4–5 times while balancing (`runs/sac_rnA_20261109_011649`, not promoted):
  after each knock-down it swung back up and caught again.
- PPO: passes only when the first catch lands. After a miss, ppo_long / B spin out, and F / G
  recover but slowly (often after 8+ s). DQN's recovery on the rig is untested.

## New tools (all opt-in; defaults unchanged)

`scripts/train_smooth_sac.py` (SAC fine-tune: reading noise, anchor), `scripts/train_rig_rl.py`
(new options `--rig-noise`, `--energy-cap`, `--arm-lag-range`, `--coupling-range`,
`--fast-arrival-speed`), `scripts/fit_sim_upright.py` (replay fit of simulator to rig logs),
`scripts/robust_gate.py` (perturbed-simulator gate), `scripts/rig_sac_online.py` (on-rig SAC
session), `scripts/tare_pendulum.py` (re-zero pendulum at rest). Further write-ups:
`docs/sac-reward-curriculum.md`, `docs/sac-improvement-proposal.md`, `docs/sim-fit-upright.md`.
