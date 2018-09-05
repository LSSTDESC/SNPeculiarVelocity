from HostGalaxies import *
import numpy
import scipy.linalg
import emcee
import pickle
import array
import os
import sys
from emcee.utils import MPIPool
import cvxopt.lapack as lapack
import cvxopt.blas as blas
from cvxopt import matrix

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
        mterm  = Deltam-M

        # if (usenp):
        #     Cinv= numpy.linalg.inv(C)
        #     logdetC = numpy.log(numpy.linalg.eigvalsh(C)).sum()
        #     lp = -0.5* (logdetC +( mterm.T @ (Cinv @ mterm) ))
        # else:
        dim = xi.shape[0]
        C_ = matrix(C)
        W = matrix(0,(dim,1),'d')
        lapack.syev(C_, W, jobz = 'N',uplo='U') 
        logdetC = sum(numpy.log(W))

        C_ = matrix(C)
        ipiv = matrix(0,(dim,1),'i')
        lapack.sytrf(C_, ipiv,uplo='U')
        lapack.sytri(C_, ipiv,uplo='U')

        mterm  = matrix(mterm)
        y = matrix(0,(dim,1),'d')

        blas.hemv(C_, mterm, y ,uplo='U')
        lp = -0.5* (logdetC +blas.dot(mterm, y) )

        if not numpy.isfinite(lp):
            return -np.inf
        return lp

    @staticmethod  
    def fit(Deltam, nsne, xi, mpi=False, p0=None, nchain=2000, **kwargs):
        ndim, nwalkers = 3, 8
        sig = numpy.array([0.1,0.01,0.01])

        if p0 is None:
            p0 = [numpy.array([1,0,0.08])+numpy.random.uniform(low = -sig, high=sig) for i in range(nwalkers)]

        if mpi:
            pool = MPIPool()
            if not pool.is_master():
                pool.wait()
                sys.exit(0)
            else:
                import time
                starttime = time.time()
                print("Start {}".format(starttime))
            sampler = emcee.EnsembleSampler(nwalkers, ndim, Fit.lnprob, args=[Deltam, nsne,xi], pool=pool)
        else:
            import time
            starttime = time.time()
            print("Start {}".format(starttime))
            sampler = emcee.EnsembleSampler(nwalkers, ndim, Fit.lnprob, args=[Deltam, nsne,xi])

        sampler.run_mcmc(p0, nchain)

        if mpi:
            if pool.is_master():
                endtime = time.time()
                print("End {}".format(endtime))
                print("Difference {}".format(endtime-starttime))
                pool.close()
        else:
            endtime = time.time()
            print("End {}".format(endtime))
            print("Difference {}".format(endtime-starttime))   

        return sampler

    @staticmethod
    def sample(galaxies,xi, **kwargs):
        m_eff =[]
        for  m, n in zip(galaxies['mB'],galaxies['nsne']):
            m_eff.append(m.sum()/n)
        m_eff=numpy.array(m_eff)
        Deltam=m_eff- galaxies['mB_expected']
        sampler = Fit.fit(m_eff- galaxies['mB_expected'],  galaxies['nsne'],xi, **kwargs)
        return sampler
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma_mu",dest="sigma_mu", default=0.08, type = float, required = False,
                    help="distance modulus standard deviation")
    parser.add_argument("--seed", dest="seed", default=1234, type = int, required = False,
                    help="random number generator seed")
    parser.add_argument('--path', dest='path', default='.', type = str, required=False)
    parser.add_argument('--frac', dest='frac', default=1, type = float, required=False)
    parser.add_argument('--savef', dest='savef', default=None, type = int, required=False)
    parser.add_argument('--nchain', dest='nchain', default=2000, type = int, required=False)
    parser.add_argument('--antecedent', dest='antecedent', default=None, type=int,required=False)
    parser.add_argument('--mpi', dest='mpi', action='store_true')
    parser.add_argument('--no-mpi', dest='mpi', action='store_false')
    parser.set_defaults(mpi=False)
    args = parser.parse_args()

    if args.savef is None:
        args.savef = args.nchain

    hg = HostGalaxies(sigma_mu=args.sigma_mu, catseed=args.seed, path=args.path)

    if args.antecedent is not None:
        chain = pickle.load(open('{}/pvlist.{}.{}.{}.pkl.{}'.format(args.path,args.sigma_mu,args.seed,args.frac,args.antecedent), "rb" ) )
    else:
        chain=None

    if (args.frac !=0):
        usehg, usexi = hg.getSubset(frac=args.frac)
    else:
        usehg = hg
        usexi = hg.xi

    for i in range(0,args.nchain//args.savef):
        if chain is None:
            initp0 = None
        else:
            initp0 = chain[:,-1,:]

        sampler = Fit.sample(usehg['galaxies'],usexi, mpi=args.mpi, nchain=args.savef, p0=initp0)

        if chain is None:
            chain=sampler.chain
        else:
            chain = numpy.concatenate((chain,sampler.chain),axis=1)

        if args.antecedent is None:
            indnm = i
        else:
            indnm = i+args.antecedent+1

        pickle.dump(chain, open('{}/pvlist.{}.{}.{}.pkl.{}'.format(args.path,args.sigma_mu,args.seed,args.frac,indnm), "wb" ) )

# python fit.py --path ../test/ --nchain 200 --savef 10
#mpirun -n 16 python fit.py --path ../out/ --mpi
#srun -n 2 -C haswell python fit.py --path ../out/
