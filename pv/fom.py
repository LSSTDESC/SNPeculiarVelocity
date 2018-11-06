#!/usr/bin/env python

import numpy as np 
import healpy as hp
import os
import re


name = 'nsn_tot.npy'
fom1=[]
fom2=[]
rt=[]
for root, dirs, files in os.walk("/Users/akim/project/lsst-cadence/zmax_eq_02/data/"):
    if name in files:
    	# to read the map 
        m = np.load(os.path.join(root, name))

    	# to display it 
    	# hp.mollview(m, nest=1)

    	# to get the total number of SNe
        nsn_tot = np.sum(m)

    	# to get the total footprint (in pixels)
        npixnon0=(m>0).sum()

    	# to get the pixel size
        npix = hp.nside2npix(64)
        omega_pix =  41253. / npix # deg^2
        sa =  omega_pix*npixnon0
        fom1.append(nsn_tot**2/sa)

        fom2.append(sa)

        dum=[m.start() for m in re.finditer('_', root)]
        rt.append(root[root.rfind('/')+5:dum[-5]])

rt = np.array(rt)

fom1=np.array(fom1)
w=np.where(rt == 'baseline2018a')[0]
fom1 = fom1 / fom1[w]

fom2=np.array(fom2)
fom2 = fom2/fom2[w]

so =np.argsort(rt)
for index in xrange(len(rt)):
    print ("{:3s}  & {:5.3f} & {:5.3f} & {:5.3f}  \\\\".format(rt[so[index]],fom1[so[index]],fom2[so[index]],(fom1[so[index]]+fom2[so[index]])/2))
# print ("{:3s} &{:8.0f} & {:5.0f}  & {:5.0f} \\".format(root[root.rfind('/')+1:][0:3], sa, nsn_tot, sa*nsn_tot/1e6))