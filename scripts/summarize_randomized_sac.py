"""Preserve the completed randomized SAC comparison and selected policy."""
import json
from pathlib import Path
import shutil
import sys

root = Path(sys.argv[1])
completion = json.loads((root / 'complete.json').read_text())
selection = json.loads((root / 'selection.json').read_text())
results = {name: json.loads((root / f'test_{name}/summary.json').read_text())
           for name in ('baseline', 'randomized')}
aggregates = {}
for name, report in results.items():
    rows = [r for r in report['results'] if r['case'].startswith('unseen_')]
    aggregates[name] = {
        'unseen_episodes': sum(r['episodes'] for r in rows),
        'unseen_successes': sum(r['successes'] for r in rows),
        'unseen_safety_stops': sum(r['safety_stops'] for r in rows),
        'unseen_max_arm_deg': max(r['max_arm_deg'] for r in rows),
        'nominal': next(r for r in report['results'] if r['case'] == 'nominal_exact'),
        'original_combined': next(r for r in report['results'] if r['case'] == 'original_combined'),
    }
output = Path('docs/results')
shutil.copy2(completion['selected_policy'], output / 'sac_randomized_policy.zip')
artifact = {'experiment': json.loads((root / 'experiment.json').read_text()),
            'continuation': json.loads((root / 'train/continuation.json').read_text()),
            'selection': selection, 'aggregates': aggregates, 'tests': results,
            'preserved_policy': 'docs/results/sac_randomized_policy.zip',
            'limitations': 'One training seed; synthetic bounds; final tests were not used for checkpoint selection. Success rates describe these sampled simulator cases only.'}
(output / 'sac_randomized_20261022.json').write_text(json.dumps(artifact, indent=2) + '\n')
print(json.dumps(aggregates, indent=2))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
colors = {'baseline': '#74859b', 'randomized': '#16806a'}
x = np.arange(20)
for offset, (name, report) in zip((-0.18, 0.18), results.items()):
    cases = [r for r in report['results'] if r['case'].startswith('unseen_')]
    axes[0].bar(x + offset, [r['successes'] for r in cases], width=0.36,
                label=name.title(), color=colors[name])
axes[0].set(xticks=x, xticklabels=[str(i + 1) for i in x], ylim=(0, 10.8),
            xlabel='Unseen physics combination', ylabel='Successful trials out of 10',
            title='Held-out swing-up and balance')
axes[0].legend(frameon=False)
for index, (name, aggregate) in enumerate(aggregates.items()):
    total = aggregate['unseen_episodes']
    count = aggregate['unseen_successes']
    axes[1].bar(index, count / total * 100, color=colors[name], width=0.55)
    axes[1].text(index, count / total * 100 + 2, f'{count}/{total}', ha='center')
axes[1].set(xticks=[0, 1], xticklabels=['Baseline', 'Randomized'], ylim=(0, 110),
            ylabel='Successful trials (%)', title='Across 20 unseen combinations')
fig.suptitle('SAC: synthetic randomized training • one training seed')
fig.tight_layout()
fig.savefig(output / 'sac_randomized_comparison.png', dpi=160)
plt.close(fig)
