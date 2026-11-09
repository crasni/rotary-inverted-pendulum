import math, sys, numpy as np, torch
torch.manual_seed(0)
f=sys.argv[1]; Z=np.load(f); D=Z['data']; ri=D[:,0].astype(int)
arm,th,armv,penv=D[:,2],D[:,3],D[:,4],D[:,5]; A=D[:,6:10]
d2=np.array([math.remainder(x,2*math.pi) for x in D[:,15]-D[:,13]])
Y=d2[:,None]*100
X=np.c_[np.sin(th),np.cos(th),penv/10,armv/5,arm,A,D[:,16],(D[:,11]-armv)/5]
held=np.isin(ri,np.arange(0,ri.max()+1,5)); up=np.abs(th)<0.5
Xl=np.c_[X,np.ones(len(X))]
c,*_=np.linalg.lstsq(Xl[~held&up],Y[~held&up],rcond=None)
v=np.var(Y[held&up]); print('upright, linear R2 held-out %.2f'%(1-np.mean((Xl[held&up]@c-Y[held&up])**2)/v))
print('  coefficients for a_k, a_k-1, a_k-2, a_k-3, a_k+1:',np.round(c[5:10,0],2),'| arm-vel gap', np.round(c[10,0],2),'| penv',np.round(c[2,0],2), '| sin', np.round(c[0,0],2))
xt=torch.tensor(X[~held],dtype=torch.float32); yt=torch.tensor(Y[~held],dtype=torch.float32)
net=torch.nn.Sequential(torch.nn.Linear(X.shape[1],64),torch.nn.Tanh(),torch.nn.Linear(64,64),torch.nn.Tanh(),torch.nn.Linear(64,1))
opt=torch.optim.Adam(net.parameters(),1e-3)
for ep in range(200):
    perm=torch.randperm(len(xt))
    for i in range(0,len(xt),1024):
        b=perm[i:i+1024]; l=((net(xt[b])-yt[b])**2).mean(); opt.zero_grad(); l.backward(); opt.step()
with torch.no_grad(): pr=net(torch.tensor(X[held],dtype=torch.float32)).numpy()
for name,m in (('upright',np.abs(th[held])<0.5),('all',np.ones(held.sum(),bool))):
    yy=Y[held][m]; print(f'  MLP held-out {name}: 40 ms theta error {np.sqrt(np.mean(yy**2))/100*57.3:.2f} -> {np.sqrt(np.mean((yy-pr[m])**2))/100*57.3:.2f} deg (R2 {1-np.mean((yy-pr[m])**2)/np.var(yy):.2f})')
