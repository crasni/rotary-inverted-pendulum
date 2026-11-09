# ppo_ecG lineage (PPO, official pass 2026-11-09)

Every step is Stable-Baselines3 PPO (`PPO.load(...).learn()`, unchanged
objective) on the lab reward plus the shaping term below. No imitation and no
other algorithm's weights. PPO is on-policy, so no logged rig transitions are
ever trained on.

| # | Checkpoint | Trained by (`scripts/train_rig_rl.py ppo ...`) |
|---|---|---|
| 1 | `runs/gpu_rl_20261016/ppo_hybrid_train` | PPO in simulation ("hybrid" = CPU rollouts, GPU optimizer; same PPO objective) |
| 2 | `runs/disturbance_rl_v2/ppo` | PPO with random pendulum disturbances |
| 3 | `docs/results/rig_rl_disturbance/ppo_long` | PPO, disturbance RMS 0–12. On the repaired rig: passes only when its first catch lands; after a miss it spins (reward exploit, below) |
| 4 | `ppo_ecB` | 400k steps: `--energy-cap 60 --energy-threshold 1.05 --fast-arrival-frac 0.3` (near-upright starts at 2–9 rad/s), delay 6–20 ms, disturbance 0–6 |
| 5 | `ppo_ecE` | 800k steps from 4: near-upright starts at 6–35 rad/s (`--fast-arrival-speed 6 35`, 40%) |
| 6 | **`ppo_ecG`** (this folder) | 1M steps from 5: delay 12–24 ms, arm lag 0–20 ms, arm-to-pendulum coupling ×1.0–1.4 per episode (`--arm-lag-range`, `--coupling-range`), 30% of episodes start from real rig states (`--reset-bank`), plus the energy cap and spin starts |

**Why the changes (to state in the report):**
- *Reward exploit.* Under the lab reward a spinning pendulum earns about 11 per tick versus 5 hanging
  (balanced: 25), and spinning faster costs almost nothing. After a missed catch, PPO stayed in
  that spin rather than brake and re-pump. The **energy cap** subtracts
  `60 * max(e - 1.05, 0)` per tick, with normalized energy `e = 0.5 (θ̇/12.3)² + cos θ`
  (θ from upright; 1 = at rest upright), so spinning at 8 rad/s costs more than hanging, while normal
  swing-ups and catches (e ≈ 1.0–1.05) are unaffected. This is training-reward shaping only. The
  evaluation metric is unchanged.
- *Start distribution.* Spin starts let PPO practise braking (on-policy PPO never meets spins
  otherwise).
- *Rig-informed randomization.* A replay fit on fast-spin segments from the repaired rig
  (`docs/sim-fit-upright.md`) matched coupling 1.2–1.4 and little arm lag; the ranges cover that.
- *Reset bank.* 6,696 near-upright observations from the 18 runs after the pendulum fix
  (all controllers' runs, excluding a hand-poked and a hand-stopped run). They are used **only as
  episode start states**. PPO learns from its own simulated rollouts after each start.

Evaluation: SAC config's arm limiter, deterministic policy, official runner, from hanging.
policy.zip sha256 087595c53b1756f294c4d78d8759a303c2910c68d82cb59113514a1ac79c3b2e
(copied unchanged from `runs/curriculum/versions/ppo_ecG`). `training.json` holds the exact settings.
