from HostGalaxies import *
import numpy
import scipy.linalg
import emcee
import pickle
import array

class Fit(object):
    """docstring for Fit"""
    def __init__(self,  sigma_mu=0.08, catseed=1234, path='../test/'):
        super(Fit, self).__init__()
        self.sigma_mu = sigma_mu
        self.catseed=catseed
        self.path=path
        self.hg = HostGalaxies(sigma_mu=sigma_mu, catseed=catseed)
        # self.xi = numpy.loadtxt('{}pvlist.{}.xi'.format(path,catseed))
        a = array.array('d')
        a.fromfile(open('{}pvlist.{}.xi','rb'),len(self.hg.galaxies['galaxy_id'])**2)
        self.xi = numpy.array(a)
        self.xi = numpy.reshape(self.xi2,(len(self.hg.galaxies['galaxy_id']),len(self.hg.galaxies['galaxy_id'])))

    @staticmethod        
    def lnprob(theta, Deltam, nsne, xi):
        A, M, sigma = theta
        if (A <=0 or sigma <0):
            return -numpy.inf
        C = A*numpy.array(xi)
        numpy.fill_diagonal(C,C.diagonal()+ sigma**2/nsne)
        Cinv= numpy.linalg.inv(C)
        logdetC = numpy.log(numpy.linalg.eigvalsh(C)).sum()
        mterm  = Deltam-M

        lp = -0.5* (logdetC +( mterm.T @ (Cinv @ mterm) ))
        if not numpy.isfinite(lp):
            return -np.inf
        return lp

    @staticmethod  
    def fit(Deltam, nsne, xi):
        ndim, nwalkers = 3, 12
        sig = numpy.array([0.1,0.01,0.01])

        p0 = [numpy.array([1,0,0.08])+numpy.random.uniform(low = -sig, high=sig) for i in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, Fit.lnprob, args=[Deltam, nsne,xi])
        sampler.run_mcmc(p0, 1000)
        return sampler

    def sample(self):
        m_eff =[]
        for  m, n in zip(self.hg.galaxies['mB'],self.hg.galaxies['nsne']):
            m_eff.append(m.sum()/n)
        m_eff=numpy.array(m_eff)
        Deltam=m_eff- self.hg.galaxies['mB_expected']
        sampler = Fit.fit(m_eff- self.hg.galaxies['mB_expected'],  self.hg.galaxies['nsne'],self.xi)
        pickle.dump(sampler.chain, open('{}pvlist.{}.{}.pkl'.format(self.path,self.sigma_mu,self.catseed), "wb" ) )

fit=Fit()
fit.sample()