"""Classify how each v3b catch (balanced >= 0.5 s) ends, for deterministic hanging starts."""
import glob, json, os, numpy as np
FILES = sorted(glob.glob('runs/curriculum/e08_v3b_hanging_det*/episode_*.npz')) + sorted(glob.glob('runs/sac_v3b_2026*/episode_000.npz'))
rows = []
for f in FILES:
    z = np.load(f, allow_pickle=True); cols = [str(c) for c in z['columns']]; x = z['samples']
    if len(x) < 300: continue
    t = x[:, 0] - x[0, 0]; arm = x[:, 1]; th = np.angle(np.exp(1j * x[:, 2])); av = x[:, 3]; pv = x[:, 4]; sent = x[:, 5]
    if 'policy_actions' in z:
        req = np.asarray(z['policy_actions'], float)
    else:
        tc = [str(c) for c in z['hardware_timing_columns']]; req = z['hardware_timing'][:, tc.index('requested_action')]
    req = np.clip(req, -1, 1)
    lim = np.abs(req - sent) > 0.02
    bal = (np.abs(th) <= np.radians(15)) & (np.abs(pv) <= 4)
    i = 0
    while i < len(bal):
        if bal[i]:
            j = i
            while j < len(bal) and bal[j]: j += 1
            dur = (j - i) / 50
            if dur >= 0.5:
                ended = j < len(bal)
                w = slice(max(i, j - 25), j)            # last 0.5 s of the hold
                hold = slice(i, j)
                drift = (arm[j - 1] - arm[i]) / max(dur, 1e-6)
                # pendulum: did theta error grow over the hold? rms of theta in last 0.5 s vs first 0.5 s
                first = slice(i, min(i + 25, j))
                grow = np.sqrt(np.mean(th[w] ** 2)) / max(np.sqrt(np.mean(th[first] ** 2)), np.radians(0.5))
                rows.append(dict(run=f.split('/')[-2][-20:] + '/' + f[-5], start=round(t[i], 2), dur=dur, ended=ended,
                                 arm_end=np.degrees(arm[j - 1]), arm_max=np.degrees(np.abs(arm[hold]).max()),
                                 drift=np.degrees(drift), lim_last=lim[w].mean(), lim_hold=lim[hold].mean(),
                                 th_end=np.degrees(th[min(j, len(th) - 1)]), pv_end=pv[min(j, len(pv) - 1)],
                                 grow=grow, av_end=av[j - 1],
                                 th_vs_arm=np.sign(th[j - 1]) == np.sign(arm[j - 1])))
            i = j
        i += 1
def kind(r):
    if not r['ended']: return 'held to end'
    if abs(r['arm_end']) >= 60 or r['lim_last'] >= 0.3: return 'arm at travel limit / limiter'
    if r['grow'] >= 3: return 'pendulum oscillation grew'
    return 'pendulum lost, arm free'
for r in rows: r['kind'] = kind(r)
print(f"{'run':26s} start  dur   end_kind                        arm_end arm_max drift°/s lim_last lim_hold grow  pv_end")
for r in rows:
    print(f"{r['run']:26s} {r['start']:5.1f} {r['dur']:5.2f}  {r['kind']:31s} {r['arm_end']:+6.0f} {r['arm_max']:6.0f} {r['drift']:+7.1f} {r['lim_last']:7.0%} {r['lim_hold']:7.0%} {r['grow']:5.1f} {r['pv_end']:+5.1f}")
import collections
for name, sel in (('all catches', rows), ('early catches (start < 5 s)', [r for r in rows if r['start'] < 5])):
    c = collections.Counter(r['kind'] for r in sel); print(name, len(sel), dict(c))
json.dump(rows, open("runs/curriculum/catch_rows.json", 'w'), default=float, indent=1)
