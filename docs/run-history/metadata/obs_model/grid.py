import sys, os, math, itertools, json
sys.path[:0] = ['src', 'scripts', 'runs/obs_model']; os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from multiprocessing import Pool
def job(args):
    delay, coupling, gain, split = args
    import replay as RP
    from train_rig_rl import RIG
    RP.SPLIT = split
    Eth, Ev, A, V0, kind = RP.replay({**RIG, 'delay_s': delay}, coupling=coupling, max_accel=gain)
    m = kind == 'sac'
    f = lambda E: float(np.degrees(np.sqrt(np.mean(E[:, 2:] ** 2))))
    return args, f(Eth), f(Eth[~m]) if (~m).any() else None, f(Eth[m]) if m.any() else None, len(Eth)
if __name__ == '__main__':
    grid = list(itertools.product((0.010, 0.018, 0.026, 0.034), (0.7, 1.0, 1.3), (0.8, 1.0, 1.25), ('train',)))
    with Pool(16) as pool:
        res = pool.map(job, grid)
    res.sort(key=lambda r: r[1])
    for a, e, ec, es, n in res[:10] + [r for r in res if r[0][:3] == (0.018, 1.0, 1.0)]:
        print(f'delay {a[0]*1e3:4.0f}ms coupling {a[1]:.1f} accel x{a[2]:.2f}: rms angle err h3-6 {e:5.2f}° (classical {ec:.2f}, SAC {es:.2f}) n={n}')
    json.dump([[list(a), e, ec, es, n] for a, e, ec, es, n in res], open('runs/obs_model/grid.json', 'w'))
