"""Per-capture hold times on the rig from the logged balance mode (hardware_timing column 6)."""
import sys, glob, json, os
import numpy as np
def holds(run):
    d = np.load(run + '/episode_000.npz', allow_pickle=True)
    mode = d['hardware_timing'][:, 6]
    out, start = [], None
    for k, m in enumerate(mode):
        if m == 1 and start is None: start = k
        if m != 1 and start is not None: out.append((k - start) / 50); start = None
    if start is not None: out.append((len(mode) - start) / 50)
    return out
def summary(name, runs):
    h = [x for r in runs for x in holds(r)]
    h = np.array(h)
    passed = sum(json.load(open(r + '/summary.json'))[0]['success'] for r in runs)
    if len(h):
        print(f'{name:28s} runs {len(runs):2d} (pass {passed})  captures {len(h):3d}  held >=1s {np.mean(h>=1):.0%}  >=2s {np.mean(h>=2):.0%}  >=5s {np.mean(h>=5):.0%}  median {np.median(h):.2f}s')
    else:
        print(f'{name:28s} runs {len(runs):2d}: no captures')
if __name__ == '__main__':
    groups = sys.argv[1:] or ['docs/results/hardware/lqr/2026*']
    for g in groups:
        runs = [r for r in sorted(glob.glob(g)) if os.path.exists(r + '/episode_000.npz')]
        runs = [r for r in runs if json.load(open(r + '/run.json'))['arguments'].get('config', '').endswith(('lqr_my_rig_coupling_delay.json',)) or 'custom' in r]
        summary(g, runs)
