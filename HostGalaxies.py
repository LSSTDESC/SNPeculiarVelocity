import pickle
import numpy

class HostGalaxies(object):
    """docstring for HostGalaxies"""

    def __init__(self, sigma_mu=0.08, catseed=1234, seed=1234):
        super(HostGalaxies, self).__init__()
        self.galaxies = pickle.load(open("tenyear.{}.{}.pkl".format(sigma_mu,seed), "rb" ))
        self.catseed=catseed
        self.sigma_mu = 0.08
        self.seed=seed
        numpy.random.seed(self.seed)

    def draganFormat(self):
        sortin = numpy.argsort(self.galaxies['redshift'])
        f = open('pvlist.{}.dat'.format(self.catseed), 'w')
        for i in range(len(sortin)):
            print(' '.join(str(e) for e in (self.galaxies['redshift'][sortin[i]],self.galaxies['mB'][sortin[i]][0],0, \
                self.galaxies['l'][sortin[i]],self.galaxies['b'][sortin[i]], self.galaxies['mB_expected'][sortin[i]], \
                self.galaxies['nsne'][sortin[i]])),file=f)
        f.close()

    def getSubsetIndeces(self,decmin=-90, decmax=90, zmax=0.2, frac=1):
        ngal = len(self.galaxies["galaxy_id"])
        arr = numpy.arange(ngal)
        numpy.random.shuffle(arr)
        w = numpy.logical_and.reduce((arr < numpy.round(ngal*frac),self.galaxies['dec'] > decmin,
            self.galaxies['dec'] < decmax,
            self.galaxies['redshift'] < zmax))
        return w

    def getSubset(self,decmin=-90, decmax=90, zmax=0.2, frac=1):
        w = self.getSubsetIndeces(decmin=decmin, decmax=decmax, zmax=zmax, frac=frac)
        out = dict()
        for key, value in self.galaxies.items():
            out[key]=value[w]
        return out