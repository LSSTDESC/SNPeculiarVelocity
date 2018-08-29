import pickle
import numpy
from chainconsumer import *

chain=  pickle.load(open("../test/pvlist.0.08.1234.pkl",'rb'))
c = ChainConsumer()
cutchain = chain[:,100:,:]
c.add_chain(numpy.reshape(cutchain,(cutchain.shape[0]*cutchain.shape[1],cutchain.shape[2])), parameters=["$A$", "$M$","$\sigma$"])
c.plotter.plot(filename="example.png", figsize="column")