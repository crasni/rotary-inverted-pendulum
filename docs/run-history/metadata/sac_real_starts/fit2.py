exec(open('runs/sac_real_starts/fit_swingup.py').read().split("print('nominal")[0])
for c in (0.9,0.95,1.0,1.05,1.1): print('coupling %.2f delay30: %.1f'%(c,err(c,1,0.03)))
for dl in (0.03,0.04,0.05,0.06,0.07): print('delay %d ms: %.1f'%(dl*1e3,err(1,1,dl)))
for dm in (0.5,0.75): print('damping x%.2f delay 40: %.1f'%(dm,err(1,dm,0.04)))
