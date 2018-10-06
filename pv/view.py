import pickle
import numpy
from chainconsumer import *

# names = ['pvlist.0.08.1234.0.2.0.2.pkl','pvlist.0.08.1234.0.3.0.2.pkl','pvlist.0.08.1234.0.5.0.2.pkl','pvlist.0.08.1234.1.0.2.pkl','pvlist.0.08.1234.1.0.1.pkl','pvlist.0.08.1234.1.0.15.pkl']
nsne = [2021,3023, 5015, 9901,1294,3870, 9901, 9901]
zmax = [0.2,0.2,0.2,0.2,0.1,0.15,0.2,0.2,0.2]
frac = [0.2,0.3,0.5,1,1,1,1,1,]
snsig = [0.08,0.08,0.08,0.08,0.08,0.08,0.1,0.12]


names = []
for fra,zma,snsi in zip(frac,zmax,snsig):
    names.append('pvlist.{}.1234.{}.{}.pkl'.format(snsi,fra,zma))


def plotter(chain):
    c = ChainConsumer()
    dum = numpy.reshape(cutchain,(cutchain.shape[0]*cutchain.shape[1],cutchain.shape[2]))
    dum[:,3] = dum[:,3]*3e5
    c.add_chain(dum, parameters=["$A$", "$M$","$\sigma_M$","$\sigma_{v}$"])
    c.plotter.plot(filename="/Users/akim/project/PeculiarVelocity/outcosmo/"+name+".png", figsize="column",truth=[None,0,0.08,None])

print("{} & {} & {} & {} & {} & {} & {} & {} \\\\".format("$z_{max}$", "fraction", "$\\sigma_{SN}$", "$N_{gal}$", \
    "$\\bar{A}$", "$\\sigma_A$", "$\\sigma_A/\\sqrt{N_{gal}}$", "$\\bar{A}/\sigma_A\\sqrt{12000/4000}$"))

for name,nsn,zma,fra,snsi in zip(names,nsne,zmax,frac,snsig):
    chain=  pickle.load(open("/Users/akim/project/PeculiarVelocity/outcosmo/"+name,'rb'))
    cutchain = chain[:,500:,:]
    # plotter(cutchain)
    # c = ChainConsumer()
    # cutchain = chain[:,500:,:]
    # dum = numpy.reshape(cutchain,(cutchain.shape[0]*cutchain.shape[1],cutchain.shape[2]))
    # dum[:,3] = dum[:,3]*3e5
    # c.add_chain(dum, parameters=["$A$", "$M$","$\sigma_M$","$\sigma_{v}$"])
    # c.plotter.plot(filename="/Users/akim/project/PeculiarVelocity/outcosmo/"+name+".png", figsize="column",truth=[None,0,0.08,None])

    print("{:4.2f} & {:4.1f} & {:4.2f} & {} & {:6.3f} & {:6.2f} & {:6.3f} & {:6.2f} \\\\".format( \
        zma,fra,snsi,nsn,cutchain[:,:,0].mean(), cutchain[:,:,0].std(), \
        cutchain[:,:,0].mean()/cutchain[:,:,0].std()/numpy.sqrt(nsn), \
        cutchain[:,:,0].mean()/cutchain[:,:,0].std()*numpy.sqrt(12000./4000)))