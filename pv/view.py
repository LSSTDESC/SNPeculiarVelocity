#!/usr/bin/env python

import pickle
import numpy
from chainconsumer import *
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
import scipy.stats

# names = ['pvlist.0.08.1234.0.2.0.2.pkl','pvlist.0.08.1234.0.3.0.2.pkl','pvlist.0.08.1234.0.5.0.2.pkl','pvlist.0.08.1234.1.0.2.pkl','pvlist.0.08.1234.1.0.1.pkl','pvlist.0.08.1234.1.0.15.pkl']

nsne = [2021,3023, 5015, 9901,1294,3870, 9901, 9901, 1215,746,563,6491, 873,2541,6491,6491, 3023, 3023, 4950]
zmax = [0.2,0.2,0.2,0.2,0.1,0.15,0.2,0.2, 0.15,0.12,0.07,0.2,0.1, 0.15,0.2, 0.2,0.2, 0.2]
frac = [0.2,0.3,0.5,1,1,1,1,1,0.3,0.3,1,0.65,0.65, 0.65, 0.65,0.65, 0.3, 0.3, 1]
snsig = [0.08,0.08,0.08,0.08,0.08,0.08,0.1,0.12,0.12,0.1,0.08,0.08,0.08, 0.08, 0.1, 0.12, 0.1, 0.12, 0.08]
tag = ["","","","","","","","","","","","","","", "skycut."]

# print(len(frac),len(zmax))

zmax_ = [0.07,0.1,0.15,0.2]
frac_ = [0.2, 0.3,0.5, 0.65, 1]
snsig_ = [0.08,0.1,0.12,0.15]

names =[]
zmax=[]
frac=[]
snsig=[]
nsne=[]
nsne_d = dict()
nsne_d[(0.2,1)]=9901
nsne_d[(0.2,0.65)]=6491
nsne_d[(0.2,0.5)]=5015
nsne_d[(0.2,0.3)]=3023
nsne_d[(0.2,0.2)]=2021

nsne_d[(0.15,1)]=3870
nsne_d[(0.15,0.65)]=2541
nsne_d[(0.15,0.5)]=1982
nsne_d[(0.15,0.3)]=1215
nsne_d[(0.15,0.2)]=800

nsne_d[(0.1,1)]=1294
nsne_d[(0.1,0.65)]=873
nsne_d[(0.1,0.5)]=683
nsne_d[(0.1,0.3)]=429
nsne_d[(0.1,0.2)]=294

nsne_d[(0.07, 1)]= 563
nsne_d[(0.07, 0.65)]=392

from pathlib import Path

for a1 in zmax_:
    for a2 in frac_:
        for a3 in snsig_:
            f = Path('/Users/akim/project/PeculiarVelocity/outcosmo/pvlist.{}.1234.{}.{}.pkl'.format(a3,a2,a1))
            print(a1, a2, a3,end='')
            if f.is_file():
                print(' exists')
                names.append('pvlist.{}.1234.{}.{}.pkl'.format(a3,a2,a1))
                zmax.append(a1)
                frac.append(a2)
                snsig.append(a3)
                nsne.append(nsne_d[(a1,a2)])
            else:
                print()


# nsne=numpy.array(nsne[:-1])
# zmax=zmax[:-1]
# frac=frac[:-1]
# snsig=snsig[:-1]
# tag=tag[:-1]

# names = []
# for fra,zma,snsi,ta in zip(frac,zmax,snsig,tag):
#     names.append('pvlist.{}.1234.{}.{}.{}pkl'.format(snsi,fra,zma,ta))

zmax = numpy.array(zmax)
frac = numpy.array(frac)
snsig = numpy.array(snsig)
nsne = numpy.array(nsne)
def plotter(chain):
    c = ChainConsumer()
    dum = numpy.reshape(cutchain,(cutchain.shape[0]*cutchain.shape[1],cutchain.shape[2]))
    dum[:,3] = dum[:,3]*3e5
    c.add_chain(dum, parameters=["$A$", "$\mathcal{M}$","$\sigma_M$","$\sigma_{v}$"])
    c.plotter.plot(filename="/Users/akim/project/PeculiarVelocity/outcosmo/"+name+".png", figsize="column",truth=[None,0,0.08,None])

print("{} & {} & {} & {} & {} & {} & {} & {} \\\\".format("$z_{max}$", "fraction", "$\\sigma_{SN}$", "$N_{gal}$", \
    "$\\bar{A}$", "$\\sigma_A$", "$\\sigma_A/\\sqrt{N_{gal}}$", "$\\bar{A}/\sigma_A\\sqrt{18000/760}$"))

effston = []
for name,nsn,zma,fra,snsi in zip(names,nsne,zmax,frac,snsig):
    chain=  pickle.load(open('/Users/akim/project/PeculiarVelocity/outcosmo/'+name,'rb'))
    cutchain = chain[:,500:,:]
    # plotter(cutchain)
    # c = ChainConsumer()
    # dum = numpy.reshape(cutchain,(cutchain.shape[0]*cutchain.shape[1],cutchain.shape[2]))
    # dum[:,3] = dum[:,3]*3e5
    # c.add_chain(dum, parameters=["$A$", "$\mathcal{M}$","$\sigma_M$","$\sigma_{v}$"])
    # c.plotter.plot(filename="/Users/akim/project/PeculiarVelocity/outcosmo/"+name+".png", figsize="column",truth=[None,0,0.08,None])
    fsigma8 = numpy.sqrt(cutchain[:,:,0])
    effston.append(fsigma8.mean()/fsigma8.std()*numpy.sqrt(18000./760))
    # print("{:4.2f} & {:4.2f} & {:4.2f} & {} & {:6.2f} & {:6.2f} & {:6.3f} & {:6.2f} \\\\".format( \
    #     zma,fra,snsi,nsn,fsigma8 .mean(), fsigma8 .std(), \
    #     fsigma8.mean()/fsigma8.std()/numpy.sqrt(nsn), \
    #     effston[-1]))
    print("{:4.2f} & {:4.2f} & {:4.2f} & {} & {:6.3f} & {:6.2f} \\\\".format( \
        zma,fra,snsi, nsn*18000./760, effston[-1], \
        effston[-1]/numpy.sqrt(18000./760*nsn)\
        ))

effston = numpy.array(effston)
mkrs=['o','s','*','D']
for z_,m in zip(zmax_,mkrs):
    fs = []
    ef = []
    for f_ in frac_:
        for sn_ in snsig_:
            w = numpy.logical_and.reduce((frac==f_ ,zmax==z_, snsig==sn_))
            if w.sum() != 0:
                fs.append(sn_**2/f_)
                ef.append(effston[w])
    fs = numpy.array(fs)*0.65/0.1**2
    plt.scatter(fs,ef,label=r"$z_{{max}}={}$".format(z_),marker=m)

plt.xlabel(r"$\frac{\sigma_M^2}{n} \left[\frac{n(10\ \mathrm{yr})}{(0.1\ \mathrm{mag})^2}\right]$",fontsize=16)
plt.ylabel(r"STON",fontsize=14)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig('notshot.png')
plt.clf()


for z_,m in zip(zmax_,mkrs):
    fs = []
    ef = []
    for f_ in frac_:
        for sn_ in snsig_:
            w = numpy.logical_and.reduce((frac==f_ ,zmax==z_, snsig==sn_))
            if w.sum() != 0:
                fs.append(sn_**2/f_)
                ef.append(effston[w]/numpy.sqrt(18000./760*nsne[w]))
    fs = numpy.array(fs)*0.65/0.1**2
    plt.scatter(fs,ef,label=r"$z_{{max}}={}$".format(z_), marker=m)

plt.xlabel(r"$\frac{\sigma_M^2}{n} \left[\frac{n(10\ \mathrm{yr})}{(0.1\ \mathrm{mag})^2}\right]$",fontsize=16)
plt.ylabel(r"STON per SN",fontsize=14)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig('perSN.png')
plt.clf()

# w = numpy.logical_and.reduce((zmax ==0.2, snsig==0.08))
# slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(frac[w],effston[w])
# plt.scatter(frac[w],effston[w])
# plt.plot([0,1],[intercept,intercept+slope])
# plt.xlabel('Frac')
# plt.ylabel('ston')
# plt.savefig('frac_.png')
# plt.clf()
volumen = FlatLambdaCDM(71,0.265).comoving_volume(0.01).value
for usez in numpy.unique(zmax):
    w = zmax ==usez
    volume = FlatLambdaCDM(71,0.265).comoving_volume(usez).value-volumen
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(1e3*nsne[w]/volume/snsig[w]**2,effston[w])
    plt.scatter(1e3*nsne[w]/volume/snsig[w]**2,effston[w],label=r"$z_{{max}}={}$".format(usez))
plt.xlabel(r'$n \sigma_M^{-2}$ $\left[10^{-3} mag^{-2} Mpc^{-3}\right]$')
plt.ylabel('Effective LSST STON')
plt.legend(loc=2)
plt.savefig('fracsnsig2_.png')
plt.clf()
# wefwe
# w = zmax ==0.2
# volume = FlatLambdaCDM(71,0.265).comoving_volume(0.2).value
# slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(1e3*nsne[w]/volume/snsig[w]**2,effston[w])
# plt.scatter(1e3*nsne[w]/volume/snsig[w]**2,effston[w])
# plt.plot([0,.7],[intercept,intercept+slope*0.7])
# plt.xlabel(r'$n \sigma^{-2}_M$ $\left[10^{-3} mag^{-2} Mpc^{-3}\right]$')
# plt.ylabel('Effective LSST STON')
# plt.savefig('fracsnsig2_.png')
# plt.clf()

# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(zmax, 1e3*nsne/volume/snsig**2,effston)
# ax.set_xlabel(r'$z_{max}$')
# ax.set_ylabel(r'$n \sigma^{-2}_M$ $\left[10^{-3} mag^{-2} Mpc^{-3}\right]$')
# ax.set_zlabel('Effective LSST STON')
# plt.savefig('3d_.png')
# plt.clf()

# wefwe
# w = numpy.logical_and.reduce((frac==1 ,snsig==0.08))
# slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(zmax[w],effston[w])
# plt.scatter(zmax[w],effston[w])
# plt.plot([0,.2],[intercept,intercept+slope*0.2])
# plt.xlabel(r'$z_{max}$')
# plt.ylabel('Effective LSST STON')
# plt.savefig('zmax_.png')
# plt.clf()

# w = numpy.logical_and.reduce((frac==1 , zmax==0.2))
# slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(snsig[w],effston[w])
# plt.scatter(snsig[w],effston[w])
# plt.plot([0,.12],[intercept,intercept+slope*0.12])
# plt.xlabel(r'$\sigma$')
# plt.ylabel('ston')
# plt.savefig('sig_.png')
# plt.clf()

# wefe

# for i in xrange(len(ston)):
#     print numpy.sqrt(numpy.sum(ston[0:i+1]**2))

# shotterm = (nsne/volume)/snsig**2
# plt.scatter(shotterm,  effston)
# plt.show()
