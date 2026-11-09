import sys, os, glob, re
sys.path.insert(0,'runs/residual'); os.environ.setdefault('OMP_NUM_THREADS','1')
from multiprocessing import Pool
import repro
if __name__=='__main__':
    cks=[]
    for v in 'MN':
        files=sorted(glob.glob(f'runs/sac_residual/{v}/checkpoints/*.zip'),key=lambda f:int(re.findall(r'(\d+)_steps',f)[0]))
        cks.append((f'{v} +300k','sac',files[len(files)//2-1],'configs/sac_my_rig.json'))
        cks.append((f'{v} final','sac',f'runs/sac_residual/{v}/policy.zip','configs/sac_my_rig.json'))
    with Pool(16) as pool:
        for n,g,ok,f,sl in pool.map(repro.job,[(*c,g) for c in cks for g in (0.0,0.8,1.0,1.3)]): print(f'{n:9s} correction {g}: pass {ok}/8 | median first arrival {f:.1f} | slow caught {sl}')
