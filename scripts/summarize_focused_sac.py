"""Preserve a completed failure-focused SAC experiment without replacing prior results."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(sys.argv[1])
completion = json.loads((root / 'complete.json').read_text())
selection = json.loads((root / 'selection.json').read_text())
reports = {name: json.loads((root / f'test_{name}/summary.json').read_text())
           for name in ('baseline', 'focused')}
diagnostics = {name: json.loads((root / f'diagnostics_{name}/summary.json').read_text())
               for name in reports}
aggregates = {}
for name, report in reports.items():
    rows = report['results']
    old = [r for r in diagnostics[name]['results'] if r['case'].startswith('previous_')]
    aggregates[name] = {
        'fresh_episodes': sum(r['episodes'] for r in rows),
        'fresh_successes': sum(r['successes'] for r in rows),
        'fresh_safety_stops': sum(r['safety_stops'] for r in rows),
        'fresh_max_arm_deg': max(r['max_arm_deg'] for r in rows),
        'previous_failure_episodes': sum(r['episodes'] for r in old),
        'previous_failure_successes': sum(r['successes'] for r in old),
        'previous_failure_safety_stops': sum(r['safety_stops'] for r in old),
        'nominal': next(r for r in diagnostics[name]['results'] if r['case'] == 'nominal_exact'),
        'original_combined': next(r for r in diagnostics[name]['results'] if r['case'] == 'original_combined'),
    }
output = Path('docs/results')
policy = output / 'sac_focused_policy.zip'
shutil.copy2(completion['selected_policy'], policy)
assert hashlib.sha256(policy.read_bytes()).hexdigest() == reports['focused']['policy_sha256']['sac']
artifact = {
    'experiment': json.loads((root / 'experiment.json').read_text()),
    'continuation': json.loads((root / 'train/continuation.json').read_text()),
    'selection': selection, 'aggregates': aggregates, 'fresh_tests': reports,
    'diagnostics': diagnostics, 'preserved_policy': str(policy),
    'completion': completion,
    'limitations': 'One continuation seed. Previous failed test combinations informed training and are diagnostics only. Fresh final combinations did not select checkpoints. Synthetic physics, no hardware validation.',
}
(output / 'sac_focused_20261022.json').write_text(json.dumps(artifact, indent=2) + '\n')
print(json.dumps(aggregates, indent=2))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.7))
colors = {'baseline': '#74859b', 'focused': '#16806a'}
x = np.arange(30)
for offset, (name, report) in zip((-0.18, 0.18), reports.items()):
    axes[0].bar(x + offset, [r['successes'] for r in report['results']],
                width=0.36, label=name.title(), color=colors[name])
axes[0].set(xticks=x[::2], xticklabels=[str(i + 1) for i in x[::2]], ylim=(0, 11),
            xlabel='Fresh physics combination', ylabel='Successes out of 10',
            title='New held-out test set')
axes[0].legend(frameon=False)
for index, (name, row) in enumerate(aggregates.items()):
    count, total = row['fresh_successes'], row['fresh_episodes']
    axes[1].bar(index, 100 * count / total, color=colors[name], width=0.55)
    axes[1].text(index, 100 * count / total + 2, f'{count}/{total}', ha='center')
axes[1].set(xticks=[0, 1], xticklabels=['Previous best', 'Selected continuation'],
            ylim=(0, 110), ylabel='Successful trials (%)', title='30 fresh combinations, 300 trials')
fig.suptitle('SAC failure-focused continuation • one continuation seed • synthetic physics')
fig.tight_layout()
fig.savefig(output / 'sac_focused_comparison.png', dpi=160)
plt.close(fig)
