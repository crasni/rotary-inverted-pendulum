import math, numpy as np, torch, sys
torch.manual_seed(0); np.random.seed(0)
Z=np.load('runs/residual/dataset.npz'); D=Z['data']; runs=Z['runs']
ri=D[:,0].astype(int); arm,th,armv,penv=D[:,2],D[:,3],D[:,4],D[:,5]; A=D[:,6:10]
th_sim,w_sim=D[:,12],D[:,13]; th_r,penv_r=D[:,15],D[:,17]
dth=np.array([math.remainder(x,2*math.pi) for x in th_r-th_sim]); dw=penv_r-w_sim
print('one-step residual rms: theta %.2f deg, pendulum velocity %.2f rad/s'%(np.degrees(np.sqrt(np.mean(dth**2))),np.sqrt(np.mean(dw**2))))
X=np.c_[np.sin(th),np.cos(th),penv/10,armv/5,arm,A, D[:,11]/5 - armv/5]
Y=np.c_[dw]
held=np.isin(ri, np.arange(0,ri.max()+1,5))   # every 5th run held out
def fit(Xtr,Ytr,Xte,Yte,hidden=64,epochs=300):
    xt=torch.tensor(Xtr,dtype=torch.float32); yt=torch.tensor(Ytr,dtype=torch.float32)
    net=torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1],hidden),torch.nn.Tanh(),torch.nn.Linear(hidden,hidden),torch.nn.Tanh(),torch.nn.Linear(hidden,Ytr.shape[1]))
    opt=torch.optim.Adam(net.parameters(),1e-3,weight_decay=1e-5)
    for ep in range(epochs):
        perm=torch.randperm(len(xt))
        for i in range(0,len(xt),1024):
            b=perm[i:i+1024]; loss=((net(xt[b])-yt[b])**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): pr=net(torch.tensor(Xte,dtype=torch.float32)).numpy()
    return net, pr
# linear baseline
Xl=np.c_[X,np.ones(len(X))]; c,*_=np.linalg.lstsq(Xl[~held],Y[~held],rcond=None); pl=Xl[held]@c
net,pr=fit(X[~held],Y[~held],X[held],Y[held])
v=np.var(Y[held])
print('held-out runs: residual explained  linear R2 %.2f | MLP R2 %.2f'%(1-np.mean((pl-Y[held])**2)/v, 1-np.mean((pr-Y[held])**2)/v))
print('linear coefficients (sin,cos,penv/10,armv/5,arm,a_k,a_k-1,a_k-2,a_k-3,armv_sim_gap):',np.round(c[:-1,0],2))
# by regime
for name,m in (('near upright |th|<0.5',np.abs(th[held])<0.5),('mid','x'),('hanging |th|>2.5',np.abs(th[held])>2.5)):
    if isinstance(m,str): m=(np.abs(th[held])>=0.5)&(np.abs(th[held])<=2.5)
    print(f'  {name:22s}: rms residual {np.sqrt(np.mean(Y[held][m]**2)):.2f} -> after MLP {np.sqrt(np.mean((Y[held][m]-pr[m])**2)):.2f} rad/s (n={m.sum()})')
torch.save(net.state_dict(),'runs/residual/mlp.pt')
