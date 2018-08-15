import pickle
import numpy
import copy

class HostGalaxies(object):
    """docstring for HostGalaxies"""

    def __init__(self, sigma_mu=0.08, catseed=1234, seed=124):
        super(HostGalaxies, self).__init__()
        self.data = pickle.load(open("../out/tenyear.{}.{}.pkl".format(sigma_mu,catseed), "rb" ))
        self.galaxies = self.data['galaxies']
        self.catseed=catseed
        self.sigma_mu = 0.08
        self.seed=seed

    def draganFormat(self, sort=False):
        if sort:
            sortin = numpy.argsort(self.galaxies['redshift'])
        else:
            sortin = numpy.arange(len(self.galaxies['redshift']))
        f = open('pvlist.{}.dat'.format(self.catseed), 'w')
        for i in range(len(sortin)):
            print(' '.join(str(e) for e in (self.galaxies['redshift'][sortin[i]],self.galaxies['mB'][sortin[i]][0],0, \
                self.galaxies['l'][sortin[i]],self.galaxies['b'][sortin[i]], self.galaxies['mB_expected'][sortin[i]])),file=f)
        f.close()

    def getSubset(self,decmin=-90, decmax=90, zmax=0.2, frac=1):
        self.galaxies['nsne'][5]=10
        self.galaxies['mB'][5]=numpy.arange(1,11)
        out = copy.deepcopy(self.data)

        # decide if supernovae are discovered or not
        if frac < 1:
            numpy.random.seed(self.seed)
            nsne = self.galaxies["nsne"].sum()
            arr = numpy.arange(nsne)
            numpy.random.shuffle(arr)       
            arr = arr < numpy.round(nsne*frac)  # array that says if SN is in frac

            sindex = 0
            for i in range(len(out['galaxies']["nsne"])):
                found = arr[sindex:sindex+self.galaxies["nsne"][i]]
                if (i==5):
                    print(found,out['galaxies']["nsne"][i],found.sum())
                out['galaxies']["nsne"][i] = found.sum()
                out['galaxies']["mB"][i] = out['galaxies']["mB"][i][found]
                sindex += self.galaxies["nsne"][i]

        w = numpy.logical_and.reduce((out['galaxies']['nsne'] > 0,out['galaxies']['dec'] > decmin,
            out['galaxies']['dec'] < decmax,
            out['galaxies']['redshift'] < zmax))

        for key, value in out["galaxies"].items():
            out["galaxies"][key]=value[w]
        return out