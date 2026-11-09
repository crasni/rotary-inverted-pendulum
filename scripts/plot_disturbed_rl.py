"""Plot the frozen, paired RL test results without selecting new policies."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, algorithm in zip(axes, ['sac', 'dqn', 'ppo']):
        report = json.loads((args.results / f'test_{algorithm}.json').read_text())
        for i, label in enumerate(['prior', 'selected']):
            cases = report['results'][f'{algorithm}_{label}']['cases']
            ax.bar(np.arange(4) + (i - .5) * .36, [c['passed'] for c in cases.values()],
                   width=.36, label=label, color=['#7f8c8d', '#2471a3'][i])
        ax.set_xticks(range(4), ['Rig', 'RMS 9', 'RMS 12', 'Combined'], rotation=20)
        ax.set_title(algorithm.upper())
        ax.set_ylim(0, report['episodes'] + 1)
        ax.set_yticks(range(0, report['episodes'] + 1, 5))
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Successful episodes / 20')
    axes[-1].legend()
    fig.suptitle('Held-out simulation: prior vs validation-selected RL policies')
    fig.tight_layout()
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
