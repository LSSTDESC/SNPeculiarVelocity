from HostGalaxies import *
import numpy
import scipy.linalg
import emcee
import pickle
import array
import os
import sys
from emcee.utils import MPIPool

class Fit(object):
    """docstring for Fit"""
    # def __init__(self):
    #     super(Fit, self).__init__()

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
        ndim, nwalkers = 3, 8
        sig = numpy.array([0.1,0.01,0.01])

        p0 = [numpy.array([1,0,0.08])+numpy.random.uniform(low = -sig, high=sig) for i in range(nwalkers)]

        pool = MPIPool()
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        else:
            import time
            starttime = time.time()
            print("Start {}".format(starttime))

        sampler = emcee.EnsembleSampler(nwalkers, ndim, Fit.lnprob, args=[Deltam, nsne,xi], pool=pool)
        # sampler = emcee.EnsembleSampler(nwalkers, ndim, Fit.lnprob, args=[Deltam, nsne,xi])
        sampler.run_mcmc(p0, 100)

        if pool.is_master():
            endtime = time.time()
            print("End {}".format(endtime))
            print("Difference {}".format(endtime-starttime))

        pool.close()
        return sampler

    @staticmethod
    def sample(galaxies,xi):
        m_eff =[]
        for  m, n in zip(galaxies['mB'],galaxies['nsne']):
            m_eff.append(m.sum()/n)
        m_eff=numpy.array(m_eff)
        Deltam=m_eff- galaxies['mB_expected']
        sampler = Fit.fit(m_eff- galaxies['mB_expected'],  galaxies['nsne'],xi)
        return sampler
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma_mu",dest="sigma_mu", default=0.08, type = float, required = False,
                    help="distance modulus standard deviation")
    parser.add_argument("--seed", dest="seed", default=1234, type = int, required = False,
                    help="random number generator seed")
    parser.add_argument('--path', dest='path', default='.', type = str, required=False)
    args = parser.parse_args()

    hg = HostGalaxies(sigma_mu=args.sigma_mu, catseed=args.seed, path=args.path)
    # hg_prune, xi = hg.getSubset(frac=0.1)
    sampler = Fit.sample(hg.galaxies,hg.xi)
    # sampler = Fit.sample(hg_prune['galaxies'],xi)
    pickle.dump(sampler.chain, open('{}/pvlist.{}.{}.pkl'.format(args.path,args.sigma_mu,args.seed), "wb" ) )

#mpirun -n 8 python fit.py --path ../out/
#srun -n 2 -C haswell python fit.py --path ../out/