import sys, os, glob, re
sys.path.insert(0,'runs/residual')
os.environ.setdefault('OMP_NUM_THREADS','1')
from multiprocessing import Pool
import repro
if __name__=='__main__':
    cks=[]
    for v in 'JKL':
        files=sorted(glob.glob(f'runs/sac_residual/{v}/checkpoints/*.zip'),key=lambda f:int(re.findall(r'(\d+)_steps',f)[0]))
        cks.append((f'{v} +250k','sac',files[len(files)//2 - 1],'configs/sac_my_rig.json'))
        cks.append((f'{v} final','sac',f'runs/sac_residual/{v}/policy.zip','configs/sac_my_rig.json'))
    jobs=[(*c,g) for c in cks for g in (0.7,1.0,1.2)]
    with Pool(15) as pool:
        for n,g,ok,f,sl in pool.map(repro.job,jobs): print(f'{n:10s} correction {g}: pass {ok}/8 | median first arrival {f:.1f} rad/s | slow arrivals caught {sl}')
