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
from scipy.linalg.lapack import *
import astropy.cosmology
from scipy.stats import cauchy
import time

class Fit(object):
    """docstring for Fit"""
    # def __init__(self):
    #     super(Fit, self).__init__()

    @staticmethod        
    def lnprob(theta, Deltam, nsne, xi,redshiftterm):
        A, M, sigma, pv = theta
        if (A <=0 or sigma <0 or pv<0):
            return -numpy.inf
        C = A*numpy.array(xi)
        numpy.fill_diagonal(C,C.diagonal()+ sigma**2/nsne + (pv*redshiftterm)**2)
        mterm  = Deltam-M

        C = matrix(C)
        W  = matrix(mterm)
        try:
            lapack.posv(C, W, uplo = 'U')
        except ArithmeticError: 
            return -np.inf
        logdetC= 2*numpy.log(numpy.array(C).diagonal()).sum()
                       
        lp = -0.5* (logdetC +blas.dot(matrix(mterm), W) ) + cauchy.logpdf(sigma, loc=0.08, scale=0.5) + cauchy.logpdf(pv, loc=0, scale=600/3e5)

        if not numpy.isfinite(lp):
            return -np.inf
        return lp

    @staticmethod  
    def fit(Deltam, nsne, xi, redshiftterm,mpi=False, p0=None, nchain=2000, **kwargs):
        ndim, nwalkers = 4, 8
        sig = numpy.array([0.1,0.01,0.01,50/3e5])

        if p0 is None:
            p0 = [numpy.array([1,0,0.08,200/3e5])+numpy.random.uniform(low = -sig, high=sig) for i in range(nwalkers)]

        if mpi:
            pool = MPIPool()
            if not pool.is_master():
                pool.wait()
                sys.exit(0)
            else:
                import time
                starttime = time.time()
                print("Start {}".format(starttime))
            sampler = emcee.EnsembleSampler(nwalkers, ndim, Fit.lnprob, args=[Deltam, nsne,xi,redshiftterm], pool=pool)
        else:
            import time
            starttime = time.time()
            print("Start {}".format(starttime))
            sampler = emcee.EnsembleSampler(nwalkers, ndim, Fit.lnprob, args=[Deltam, nsne,xi,redshiftterm])

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
        sampler = Fit.fit(m_eff- galaxies['mB_expected'],  galaxies['nsne'],xi, \
            5/numpy.log(10)*(1+galaxies['redshift_true'])/galaxies['redshift_true'],**kwargs)
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
    parser.add_argument('--zmax', dest='zmax', default=0.2, type=float,required=False)
    parser.add_argument('--skycut', dest='skycut', action='store_true')
    parser.add_argument('--no-skycut', dest='skycut', action='store_false')
    parser.set_defaults(skycut=False)
    parser.add_argument('--mpi', dest='mpi', action='store_true')
    parser.add_argument('--no-mpi', dest='mpi', action='store_false')
    parser.set_defaults(mpi=False)
    args = parser.parse_args()

    if args.savef is None:
        args.savef = args.nchain

    if args.skycut:
        decmax=-35.65
        bmin=-90.
        bmax=90
        fstr='skycut.'
    else:
        decmax=90
        bmin=-90
        bmax=90
        fstr=''

    hg = HostGalaxies(sigma_mu=args.sigma_mu, catseed=args.seed, path=args.path)

    if args.antecedent is not None:
        chain = pickle.load(open('{}/pvlist.{}.{}.{}.{}.{}pkl.{}'.format(args.path,args.sigma_mu,args.seed,args.frac,args.zmax,fstr,args.antecedent), "rb" ) )
    else:
        chain=None

    if (args.frac !=1 or args.zmax !=0.2 or args.skycut):
        usehg, usexi = hg.getSubset(frac=args.frac, zmax=args.zmax,decmax=decmax, bmin=bmin, bmax=bmax)
    # elif (args.zmax !=0.2):
    #     usehg, usexi = hg.getSubset(zmax=args.zmax, decmax=decmax, bmin=bmin, bmax=bmax)
    #     print(usexi.shape)
    # elif (args.skycut)
        print (usexi.shape)

    else:
        usehg = hg.data
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

        pickle.dump(chain, open('{}/pvlist.{}.{}.{}.{}.{}pkl.{}'.format(args.path,args.sigma_mu,args.seed,args.frac,args.zmax,fstr,chain.shape[1]), "wb" ) )

#srun -n 1 -c 64 --cpu-bind=sockets python fit.py --path ../out/ --frac 0.5  --nchain 2
#python fit.py --path ../out/ --nchain 1 frac 0.19
# python fit.py --path ../test/ --nchain 200 --savef 10
#mpirun -n 16 python fit.py --path ../out/ --mpi
