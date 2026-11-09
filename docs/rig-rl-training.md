# RL fine-tuning on the identified rig model

These are standalone policy experiments in simulation. The LQR hardware-pass tag uses SAC for swing-up and LQR for balance. Its hardware result does not measure standalone SAC performance.

Training starts from the saved PPO, DQN and randomized SAC weights. It loads the system-identification file, sets simulated command delay to 18 ms and planner speed to 4.4 rad/s, and applies the command limiter in `configs/sac_my_rig.json`. Resets are 25% exact hanging, 25% near upright, and 50% varied hanging. DQN and SAC collect 5,000 new transitions before updating. Each run uses CPU and seed 73.

Run from the branch checkout with the repository's RL dependencies installed.

```bash
export PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/train_rig_rl.py ppo --steps 65536 --output runs/rig_rl/ppo
python scripts/train_rig_rl.py dqn --steps 75000 --output runs/rig_rl/dqn
python scripts/train_rig_rl.py sac --steps 75000 --output runs/rig_rl/sac
python scripts/eval_rig_rl.py --candidates runs/rig_rl --output runs/rig_rl/comparison.json
```

The paired evaluation applies the same limiter to all policies and uses the assignment success metric over 15-second episodes. It evaluates one deterministic hanging start, ten varied hanging starts, and ten varied starts under the existing combined synthetic stress case. The latter varies friction, inertia, delay, noise, acceleration and planner speed together. These scenarios and one fine-tuning seed give a simulation comparison. Performance on the physical rig remains unmeasured for these checkpoints.

| Policy | Exact hanging | Varied hanging | Combined stress |
| --- | ---: | ---: | ---: |
| PPO baseline → fine-tuned | 0/1 → 0/1 | 0/10 → 0/10 | 1/10 → 0/10 |
| DQN baseline → fine-tuned | 1/1 → 1/1 | 3/10 → 10/10 | 7/10 → 8/10 |
| Randomized SAC baseline → fine-tuned | 1/1 → 1/1 | 10/10 → 10/10 | 1/10 → 10/10 |

No run triggered the simulator arm guard. Fine-tuned SAC reached 80.7° of arm travel in the combined case, compared with 53.9° for its baseline.

The [comparison JSON](results/rig_rl/comparison.json) includes the focused SAC baseline and checkpoint hashes. The fine-tuned checkpoints are [SAC](results/rig_rl/sac/policy.zip), [DQN](results/rig_rl/dqn/policy.zip) with its [action grid](results/rig_rl/dqn/policy.zip.actions.json), and [PPO](results/rig_rl/ppo/policy.zip).
