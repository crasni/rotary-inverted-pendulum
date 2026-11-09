"""Continue SAC on previously failed physics; evaluate a fresh independent set."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .contracts import PARAMS, manifest
from .randomized_training import RandomizedPhysics, RANGES
from .robustness import CASES, run_suite


class FailureSampler:
    """25% exact anchors, 40% local perturbations, 35% broad randomization."""
    def __init__(self, previous_report):
        rows = previous_report['tests']['randomized']['results']
        self.failures = [r for r in rows if r['case'].startswith('unseen_')
                         and r['successes'] < r['episodes']]
        if not self.failures:
            raise ValueError('No failed combinations in previous report')
        weights = np.array([r['episodes'] - r['successes'] for r in self.failures], dtype=float)
        self.weights = weights / weights.sum()

    def local(self, rng):
        anchor = self.failures[int(rng.choice(len(self.failures), p=self.weights))]
        case = {key: float(np.clip(rng.normal(anchor['settings'][key], 0.08 * (high - low)),
                                   low, high)) for key, (low, high) in RANGES.items()}
        return case, anchor['case']

    def sample(self, rng):
        choice = rng.random()
        if choice < 0.25:
            anchor = self.failures[int(rng.choice(len(self.failures), p=self.weights))]
            case = {key: anchor['settings'][key] for key in RANGES}
            source, label = 'exact_failure', anchor['case']
        elif choice < 0.65:
            case, label = self.local(rng)
            source = 'near_failure'
        else:
            case = {key: float(rng.uniform(*bounds)) for key, bounds in RANGES.items()}
            source, label = 'broad', None
        return {**case, 'sampling_source': source, 'failure_anchor': label}


class FocusedPhysics(RandomizedPhysics):
    def __init__(self, sampler, **kwargs):
        self.sampler = sampler
        super().__init__(**kwargs)

    def sample_case(self):
        return self.sampler.sample(self.rng)

    def configure_plant(self, plant):
        # 20% upright, 10% exact hanging, 70% perturbed hanging.
        plant.upright_reset_frac = 0.2
        plant.exact_hanging_reset_frac = 0.1


def validation_cases(sampler):
    rng = np.random.default_rng(181000)
    cases = {f'validation_broad_{i:02d}': {'varied_start': True, **{
        key: float(rng.uniform(*bounds)) for key, bounds in RANGES.items()}}
        for i in range(6)}
    for i in range(6):
        case, _ = sampler.local(rng)
        cases[f'validation_near_failure_{i:02d}'] = {'varied_start': True, **case}
    return cases


def select_score(report):
    rows = report['results']
    # Prefer zero/fewer guard activations, then balancing success.
    return (-sum(r['safety_stops'] for r in rows),
            sum(r['successes'] for r in rows),
            float(np.mean([r['mean_longest_balance_s'] for r in rows])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/sac_focused_gpu.json'))
    args = parser.parse_args()
    import torch
    from .algorithms.sac import train
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this experiment')
    torch.set_num_threads(1)
    source = Path('docs/results/sac_randomized_20261022.json')
    previous = json.loads(source.read_text())
    sampler = FailureSampler(previous)
    config = json.loads(args.config.read_text())
    baseline = Path(config['resume_checkpoint'])
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {
        'contract': manifest(PARAMS), 'gpu': torch.cuda.get_device_name(0),
        'config': config, 'ranges': RANGES, 'source_report': str(source),
        'source_report_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'baseline_sha256': hashlib.sha256(baseline.read_bytes()).hexdigest(),
        'failure_anchors': sampler.failures, 'anchor_weights': sampler.weights.tolist(),
        'sampling_probabilities': {'exact_failure': 0.25, 'near_failure': 0.40, 'broad': 0.35},
        'local_std_fraction_of_range': 0.08,
        'reset_probabilities': {'upright': 0.2, 'exact_hanging': 0.1, 'perturbed_hanging': 0.7},
        'validation_seed': 181000, 'test_seed': 281000, 'diagnostic_seed': 381000,
        'selection': 'fewest safety stops, most successes, longest mean balance; best new candidate tested, baseline eligible for recommendation',
        'limitations': 'Previous test cases now inform training. Only the fresh test set remains held out. One continuation seed; synthetic physics; no hardware validation.',
    }
    (args.output / 'experiment.json').write_text(json.dumps(metadata, indent=2) + '\n')
    train(env_factory=lambda: FocusedPhysics(sampler, log_path=args.output / 'physics.jsonl'),
          config=config, output_dir=args.output / 'train')
    candidates = [baseline, *sorted((args.output / 'train/checkpoints').glob('*.zip')),
                  args.output / 'train/policy.zip']
    scores = []
    for index, policy in enumerate(candidates):
        report = run_suite({'sac': policy}, params=PARAMS,
                           output=args.output / f'validation_{index:02d}',
                           episodes=5, seed=181000, cases=validation_cases(sampler))
        score = select_score(report)
        scores.append({'policy': str(policy), 'safety_stops': -score[0],
                       'successes': score[1], 'balance': score[2]})
    rank = lambda r: (-r['safety_stops'], r['successes'], r['balance'])
    recommended = max(scores, key=rank)
    best = max(scores[1:], key=rank)
    (args.output / 'selection.json').write_text(json.dumps({
        'scores': scores, 'selected': best, 'recommended': recommended}, indent=2) + '\n')
    rng = np.random.default_rng(281000)
    unseen = {f'unseen_{i:02d}': {'varied_start': True, **{
        key: float(rng.uniform(*bounds)) for key, bounds in RANGES.items()}}
        for i in range(30)}
    diagnostics = {f'previous_{r["case"]}': r['settings'] for r in sampler.failures}
    diagnostics.update(nominal_exact={}, original_combined=CASES['combined'])
    for label, policy in [('baseline', baseline), ('focused', Path(best['policy']))]:
        run_suite({'sac': policy}, params=PARAMS, output=args.output / f'test_{label}',
                  episodes=10, seed=281000, cases=unseen)
        run_suite({'sac': policy}, params=PARAMS, output=args.output / f'diagnostics_{label}',
                  episodes=10, seed=381000, cases=diagnostics)
    (args.output / 'complete.json').write_text(json.dumps({
        'selected_policy': best['policy'], 'recommended_policy': recommended['policy'],
        'recommended_baseline': recommended['policy'] == str(baseline),
        'note': 'Fresh test was not used for checkpoint selection. Diagnostics are on physics used for training.'
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
