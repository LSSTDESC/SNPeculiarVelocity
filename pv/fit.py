from HostGalaxies import *
import numpy
import scipy.linalg
import emcee

hg = HostGalaxies(path='../test/')
xi = numpy.loadtxt("../test/pvlist.1234.xi")

m_eff =[]
for  m, n in zip(hg.galaxies['mB'],hg.galaxies['nsne']):
    m_eff.append(m.sum()/n)
m_eff=numpy.array(m_eff)

#immutables that enter the calculation
ngal = len(hg.galaxies['mB'])
Deltam=m_eff- hg.galaxies['mB_expected']

def lnprob(theta):
    A, M, sigma = theta
    C = A*numpy.array(xi)
    numpy.fill_diagonal(C,C.diagonal()+ sigma**2/hg.galaxies['nsne'])
    Cinv= numpy.linalg.inv(C)
    Croots = scipy.linalg.sqrtm(C)
    logdetC = 2*numpy.log(Croots.diagonal()).sum()

    mterm  = Deltam-M

    lp = -logdetC - 0.5* ( mterm.T @ (Cinv @ mterm) )
    if not numpy.isfinite(lp):
        return -np.inf
    return lp

ndim, nwalkers = 3, 10
sig = numpy.array([0.1,0.01,0.01])

p0 = [numpy.array([1,0,0.1])+numpy.random.uniform(low = -sig, high=sig) for i in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
sampler.run_mcmc(p0, 100)

import pickle
pickle.dump(sampler.flatchain, open( "emcee.pkl", "wb" ) )