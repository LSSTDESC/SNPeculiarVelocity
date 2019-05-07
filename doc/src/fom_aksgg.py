#!/usr/bin/env python

import numpy as np 
import matplotlib.pyplot as plt
#import healpy as hp
import os
import re
import partials

name = 'nsn_tot.npy'
skyfrac=[]
rt=[]
for root, dirs, files in os.walk("Footprints/"): # for 10 years footprints
    if name in files:
    	# to read the map 
        m = np.load(os.path.join(root, name))

    	# to display it 
    	# hp.mollview(m, nest=1)

    	# to get the total number of SNe
        nsn_tot = np.sum(m)

    	# to get the total footprint (in pixels)
        npixnon0=(m>0).sum()

        # get the total number of pixels
        npixtot = npixnon0 + (m==0).sum()

        if root[root.rfind('/')+1:] == 'baseline2018a' :
            skyfrac_ref =  npixnon0/npixtot

        else:

            skyfrac.append(npixnon0/npixtot)

            dum=[m.start() for m in re.finditer('_', root)]
            rt.append(root[root.rfind('/')+1:])

rt = np.array(rt)
#print('rt',rt)

skyfrac=np.array(skyfrac)
#print("skyfrac",skyfrac)

#w=np.where(rt == 'baseline2018a')[0]
#skyfrac_ref= skyfrac[w]
#print(skyfrac_ref)

fom_ref=partials.surveyFOM(skyfrac_ref,10)
#print("fom_ref",fom_ref)
fom_10 = partials.surveyFOM(skyfrac,10) / fom_ref
#print(fom_10)

rt=rt

so =np.argsort(rt)
dtp = []
list = []

for index in range(len(rt)):
    list= (so[index],fom_10[so[index]])
    dtp.append((list))

#print("liste=",dtp)
dtype = [('cadence_argsort', float), ('fom_10', float)]
to_sort = np.array(dtp, dtype=dtype)
to_plot = np.sort(to_sort,order="fom_10") 

tp_cadence = []
tp_fom_10 = [] 

for index in range(len(to_plot)):
    i = len(to_plot) - index
    tp_cadence.append(rt[int(to_plot[i][0])])
    tp_fom_10.append(to_plot[i][1])

print("Footprints : ", tp_cadence)
print("FoM over 10 years : ", tp_fom_10)

#fig = plt.figure(figsize=(10,15))
#ax = fig.add_subplot(1, 1, 1)
#plt.barh(tp_cadence,tp_fom_10, alpha=0.7, color="grey")
#plt.hlines(1, 0, 10, colors='k', linestyles='-', label='baseline2018a')
#plt.xticks(rotation=0,fontsize=20)
#plt.yticks(fontsize=20)
#plt.tick_params(length=10,width=1.2)
#plt.xlim(-0.5,12.5)
#plt.title('Gamma FoM ',fontsize=25)
#plt.savefig('FOM_Gamma_OSTF.pdf')
#plt.show()
