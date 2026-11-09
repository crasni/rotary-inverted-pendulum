# rnA lineage (reward-only SAC, official passes 2026-11-09)

Every step below is Stable-Baselines3 SAC with its ordinary losses (critic,
actor, automatic entropy temperature) and the lab's reward. Checked from each
step's metadata and the training code. Oldest first.

| # | Checkpoint | Trained by | Data | Non-standard parts |
|---|---|---|---|---|
| 1 | `docs/results/sac_randomized_policy.zip` | SAC in the randomized simulator (`sac_randomized_20261022.json`) | simulator | – |
| 2 | `docs/results/rig_rl/sac` | SAC fine-tune on the identified rig model, 75k steps (`train_rig_rl.py`) | simulator | – |
| 3 | `runs/disturbance_rl_v2/sac` | SAC with random pendulum disturbances | simulator | – |
| 4 | `docs/results/rig_rl_disturbance/sac_long` | SAC, disturbance RMS 0–12, 300k steps | simulator | – |
| 5 | `runs/sac_finetune/V1` | SAC, 600k steps, velocity-reading noise 0.4–1.2 rad/s (`finetune_real.py`) | simulator only: `real_fraction` 0, so the listed LQR/iLQR hardware runs were never in the replay buffer; they were only used to *monitor* TD error under `torch.no_grad()` | – |
| 6 | `curriculum/versions/v2m2` | SAC, 60k simulator steps, each batch 50% on-rig ticks (`rig_curriculum.py update`) | simulator + SAC's own rig ticks (V1 episodes `b03`–`b05`, 4,823 transitions) | `b03`–`b05` episodes began from an LQR placement; LQR's ticks were never saved as transitions |
| 7 | `curriculum/versions/v3b` | as 6 | + SAC's own rig ticks from v2m2 (`e05`, `e06`, `b06`), only ticks within 45° of upright | – |
| 8 | `curriculum/versions/v6b` | as 6, 40k steps | + SAC rig ticks from v3, v3b, v5 (`e07`–`e09`, `b07`, `b08`); catch ticks ×2 | **swing-up anchor**: extra actor loss pulling actions on SAC's own rig swing-up states (≥ 30° from upright) toward the parent's (v3b) |
| 9 | `curriculum/versions/v6b_runner` = `sac_v6b_reward_only` | none (re-saved for the runner) | – | weights identical to v6b |
| 10 | **`curriculum/versions/rnA`** = this folder | SAC, 200k steps in V1's training simulator (`train_smooth_sac.py --rig-noise 1`) | simulator, with rig-like heavy-tailed reading noise (Student-t, 0.6° / 0.6 rad/s rms) | **swing-up anchor** to v6b on SAC's own rig swing-up states (≥ 45°). `rnA_training.json` says "no rig data": it means no rig transitions or rewards; the anchor does use those rig *states* |

Not anywhere in the chain: P1 or any imitation (`rig_curriculum.py` refuses a
checkpoint whose history contains `pretraining.json`), LQR or other controllers'
commands, other controllers' transitions in any replay buffer.

To disclose in the report: (a) steps 6–8 learned partly from SAC's own rig
experience; (b) some of those episodes started from an LQR placement (a reset
tool, not training data); (c) the swing-up anchor is an addition to SAC's loss
(a trust region on SAC's own earlier policy). The graded runs are SAC alone,
deterministic, from hanging, through the official runner.
