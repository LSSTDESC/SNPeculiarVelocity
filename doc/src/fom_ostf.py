 #!/usr/bin/env python

import numpy as np 
import matplotlib.pyplot as plt

tp_cadence=['rolling_mix_10yrs_opsim', 'pontus_2489', 'rolling_10yrs_opsim', 'kraken_2042', 'alt_sched_rolling', 'kraken_2044', 'kraken_2026', 'kraken_2035', 'pontus_2002', 'colossus_2667', 'mothra_2049', 'nexus_2097', 'pontus_2573', 'alt_sched'] 

tp_fom_10=[0.82523797,0.86024946,0.99581343,0.97263434,0.90679374,0.71793979,0.99543942,1.01288421,0.72109293,0.82840686,0.71807872,0.71547569,0.73722483,0.91633328] 

idx=[11,5,10,8,12,0,9,1,4,13,3,6,2,7] 
p1=[]
p2=[]

for i in idx: 
	p1.append(tp_cadence[i])
	p2.append(tp_fom_10[i])

fig = plt.figure(figsize=(10,15))
ax = fig.add_subplot(1, 1, 1)
#plt.grid()
plt.barh(p1,p2, alpha=0.65, color="black")
plt.xticks(rotation=0,fontsize=20)
plt.yticks(fontsize=20)
plt.tick_params(length=10,width=1.2)
plt.xlabel("$\sigma_\gamma$(cadence) / $\sigma_\gamma$(baseline2018a)",fontsize=20)
plt.ylim(0.5,13.5)
plt.show()
plt.savefig('FOM_Gamma_OSTF_new.pdf')
