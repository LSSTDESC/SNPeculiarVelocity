import pickle
import numpy
from chainconsumer import *

# names = ['pvlist.0.08.1234.0.2.0.2.pkl','pvlist.0.08.1234.0.3.0.2.pkl','pvlist.0.08.1234.0.5.0.2.pkl','pvlist.0.08.1234.1.0.2.pkl','pvlist.0.08.1234.1.0.1.pkl','pvlist.0.08.1234.1.0.15.pkl']
nsne = [2021,3023, 5015, 9901,1294,3870]
zmax = [0.2,0.2,0.2,0.2,0.1,0.15]
frac = [0.2,0.3,0.5,1,1,1]

names = []
for fra,zma in zip(frac,zmax):
    names.append('pvlist.0.08.1234.{}.{}.pkl'.format(fra,zma))


def plotter(chain):
    c = ChainConsumer()
    dum = numpy.reshape(cutchain,(cutchain.shape[0]*cutchain.shape[1],cutchain.shape[2]))
    dum[:,3] = dum[:,3]*3e5
    c.add_chain(dum, parameters=["$A$", "$M$","$\sigma_M$","$\sigma_{v}$"])
    c.plotter.plot(filename="/Users/akim/project/PeculiarVelocity/outcosmo/"+name+".png", figsize="column",truth=[None,0,0.08,None])

for name,nsn,zma,fra in zip(names,nsne,zmax,frac):
    chain=  pickle.load(open("/Users/akim/project/PeculiarVelocity/outcosmo/"+name,'rb'))
    cutchain = chain[:,500:,:]
    # plotter(cutchain)
    # c = ChainConsumer()
    # cutchain = chain[:,500:,:]
    # dum = numpy.reshape(cutchain,(cutchain.shape[0]*cutchain.shape[1],cutchain.shape[2]))
    # dum[:,3] = dum[:,3]*3e5
    # c.add_chain(dum, parameters=["$A$", "$M$","$\sigma_M$","$\sigma_{v}$"])
    # c.plotter.plot(filename="/Users/akim/project/PeculiarVelocity/outcosmo/"+name+".png", figsize="column",truth=[None,0,0.08,None])

    print("{:4.2f} & {:4.1f} & {} & {:6.3f} & {:6.2f} & {:6.2f} & {:6.3f}  \\\\ \n".format( \
        zma,fra,nsn,cutchain[:,:,0].mean(), cutchain[:,:,0].std(), \
        cutchain[:,:,0].mean()/cutchain[:,:,0].std()*numpy.sqrt(12000./4000), \
        cutchain[:,:,0].mean()/cutchain[:,:,0].std()/numpy.sqrt(nsn)))