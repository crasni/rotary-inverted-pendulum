import math, sys, numpy as np, torch
sys.path[:0]=['src']
from pendulum_lab.residual import network
torch.manual_seed(0)
Z=np.load('runs/residual/dataset2_10.npz'); D=Z['data']; ri=D[:,0].astype(int)
arm,th,armv,penv=D[:,2],D[:,3],D[:,4],D[:,5]; A=D[:,6:10]
d2=np.array([math.remainder(x,2*math.pi) for x in D[:,15]-D[:,13]])
acc=np.clip(2*d2/0.04**2,-150,150)            # rad/s^2 that explains the 40 ms error
SCALE=50.0
X=np.c_[np.sin(th),np.cos(th),penv/10,armv/5,arm,A].astype(np.float32); Y=(acc/SCALE)[:,None].astype(np.float32)
held=np.isin(ri,np.arange(0,ri.max()+1,5))
near=np.abs(th)<1.0
mode=sys.argv[1] if len(sys.argv)>1 else 'heldout'
tr=(~held if mode=='heldout' else np.ones(len(X),bool))&near
net=network(64); opt=torch.optim.Adam(net.parameters(),1e-3,weight_decay=1e-5)
xt=torch.tensor(X[tr]); yt=torch.tensor(Y[tr])
for ep in range(250):
    perm=torch.randperm(len(xt))
    for i in range(0,len(xt),1024):
        b=perm[i:i+1024]; l=(torch.nn.functional.huber_loss(net(xt[b]),yt[b],delta=1.0)); opt.zero_grad(); l.backward(); opt.step()
with torch.no_grad(): pr=net(torch.tensor(X[held&near])).numpy()
print('held-out accel residual: rms %.1f -> %.1f rad/s^2'%(np.sqrt(np.mean((Y[held&near]*SCALE)**2)),np.sqrt(np.mean(((Y[held&near]-pr)*SCALE)**2))))
torch.save({'state':net.state_dict(),'hidden':64,'accel_scale':SCALE,'trained_on':mode,'held_out_runs':[str(r) for r in Z['runs'][np.unique(ri[held])]]},f'runs/residual/residual_{mode}.pt')
