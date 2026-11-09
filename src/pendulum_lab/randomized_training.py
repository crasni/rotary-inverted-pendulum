"""Episode-wise synthetic randomization and a reproducible GPU SAC experiment."""
import argparse
import json
from pathlib import Path
import tempfile

import gymnasium as gym
import numpy as np

from .contracts import PARAMS, manifest
from .environment import make_env
from .robustness import ObservationNoise, varied_parameters, run_suite

# Provisional hypotheses, not measured hardware confidence intervals.
RANGES = {
    'friction_factor': [0.5, 1.5], 'inertia_factor': [0.8, 1.2],
    'delay_s': [0.010, 0.030], 'max_accel': [40.0, 50.0],
    'speed_limit': [4.0, 5.0], 'angle_noise': [0.0, 0.010],
    'velocity_noise': [0.0, 0.3],
}


class RandomizedPhysics(gym.Env):
    """Rebuild the plant at reset; keep physics fixed within each episode."""
    metadata = {}

    def __init__(self, params=PARAMS, log_path=None):
        self.params = Path(params)
        self.temp = tempfile.TemporaryDirectory(prefix='pendulum-randomized-')
        self.rng = np.random.default_rng()
        self.env = make_env(self.params, start='mixed')
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.log_path = Path(log_path) if log_path else None

    def sample_case(self):
        return {key: float(self.rng.uniform(*bounds)) for key, bounds in RANGES.items()}

    def configure_plant(self, plant):
        """Optional experiment-specific reset distribution."""

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        case = self.sample_case()
        episode_seed = int(self.rng.integers(0, 2**31 - 1))
        path = varied_parameters(self.params, case, Path(self.temp.name) / 'params.json')
        self.env.close()
        self.env = make_env(path, start='mixed')
        plant = self.env.unwrapped
        plant._fixed_transport_delay_s = case['delay_s']
        plant._fixed_motor_max_accel_rad_s2 = case['max_accel']
        plant.motor_planner_speed_limit_rad_s = case['speed_limit']
        self.configure_plant(plant)
        self.env = ObservationNoise(self.env, case['angle_noise'], case['velocity_noise'])
        obs, info = self.env.reset(seed=episode_seed, options=options)
        if self.log_path:
            with self.log_path.open('a') as stream:
                stream.write(json.dumps({'seed': episode_seed, **case}) + '\n')
        info['randomized_physics'] = case
        return obs, info

    def step(self, action):
        return self.env.step(action)

    def close(self):
        self.env.close()
        self.temp.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=600000)
    args = parser.parse_args()
    import torch
    from .algorithms.sac import train
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this experiment')
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    baseline = Path('runs/sac_robustness_20261016/train/policy.zip')
    config = json.loads(Path('configs/sac_randomized_gpu.json').read_text())
    config.update(seed=73, total_steps=args.steps, resume_checkpoint=str(baseline),
                  checkpoint_every=100000, save_replay_buffer=False)
    (args.output / 'experiment.json').write_text(json.dumps({
        'contract': manifest(PARAMS), 'gpu': torch.cuda.get_device_name(0),
        'training_ranges': RANGES, 'config': config,
        'training_start': 'mixed', 'validation_seed': 51000, 'test_seed': 91000,
        'limitations': 'Synthetic ranges; one training seed; no hardware validation.'
    }, indent=2) + '\n')
    train(env_factory=lambda: RandomizedPhysics(log_path=args.output / 'physics.jsonl'),
          config=config, output_dir=args.output / 'train')
    # Validation selects a checkpoint. The later test uses independent combinations
    # and seeds, and is never used for selection.
    rng = np.random.default_rng(51000)
    validation = {f'validation_{i:02d}': {'varied_start': True, **{
        k: float(rng.uniform(*v)) for k, v in RANGES.items()}} for i in range(8)}
    candidates = [baseline, *sorted((args.output / 'train/checkpoints').glob('*.zip')),
                  args.output / 'train/policy.zip']
    scores = []
    for index, policy in enumerate(candidates):
        report = run_suite({'sac': policy}, params=PARAMS,
                           output=args.output / f'validation_{index:02d}',
                           episodes=3, seed=51000, cases=validation)
        rows = report['results']
        scores.append({'policy': str(policy), 'successes': sum(r['successes'] for r in rows),
                       'safety_stops': sum(r['safety_stops'] for r in rows),
                       'balance': float(np.mean([r['mean_longest_balance_s'] for r in rows]))})
    # Always select among newly trained checkpoints; baseline is a comparison.
    best = max(scores[1:], key=lambda r: (r['successes'], -r['safety_stops'], r['balance']))
    (args.output / 'selection.json').write_text(json.dumps({'scores': scores, 'selected': best}, indent=2))
    rng = np.random.default_rng(91000)
    unseen = {f'unseen_{i:02d}': {'varied_start': True, **{
        k: float(rng.uniform(*v)) for k, v in RANGES.items()}} for i in range(20)}
    unseen['nominal_exact'] = {}
    from .robustness import CASES
    unseen['original_combined'] = CASES['combined']
    for label, policy in [('baseline', baseline), ('randomized', Path(best['policy']))]:
        run_suite({'sac': policy}, params=PARAMS, output=args.output / f'test_{label}',
                  episodes=10, seed=91000, cases=unseen)
    (args.output / 'complete.json').write_text(json.dumps({'selected_policy': best['policy'],
        'baseline_summary': str(args.output / 'test_baseline/summary.json'),
        'randomized_summary': str(args.output / 'test_randomized/summary.json')}, indent=2))


if __name__ == '__main__':
    main()
