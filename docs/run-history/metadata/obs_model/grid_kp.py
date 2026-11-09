import sys, os, math, itertools, json
sys.path[:0] = ['src', 'scripts', 'runs/obs_model']; os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from multiprocessing import Pool
def job(args):
    kp, zeta, delay, split = args
    import replay as RP
    from train_rig_rl import RIG
    RP.SPLIT = split
    Eth, Ev, A, V0, kind = RP.replay({**RIG, 'delay_s': delay}, kp=kp, zeta=zeta)
    Earm = np.array(RP.EARM); m = kind == 'sac'
    f = lambda E: float(np.degrees(np.sqrt(np.mean(E[:, 2:] ** 2))))
    return args, f(Eth), f(Eth[~m]) if (~m).any() else float('nan'), f(Eth[m]) if m.any() else float('nan'), f(Earm), len(Eth)
if __name__ == '__main__':
    split = sys.argv[1] if len(sys.argv) > 1 else 'train'
    if split == 'train':
        grid = [(1000, 1.0, 0.018, 'train')] + list(itertools.product((60, 25, 12, 6, 3), (0.05, 0.15, 0.4, 1.0), (0.018,), ('train',)))
    else:
        grid = [tuple(json.loads(x)) + (split,) for x in sys.argv[2:]]
    with Pool(16) as pool:
        res = pool.map(job, grid)
    res.sort(key=lambda r: r[1])
    for a, e, ec, es, ea, n in res:
        fn = math.sqrt(a[0] / 1.065625e-4) / (2 * math.pi)
        print(f'kp {a[0]:5.0f} (~{fn:4.0f} Hz) zeta {a[1]:.2f} delay {a[2]*1e3:.0f}ms [{a[3]}]: pendulum err h3-6 {e:5.2f}° (classical {ec:.2f}, SAC {es:.2f}) arm err {ea:.2f}° n={n}', flush=True)
