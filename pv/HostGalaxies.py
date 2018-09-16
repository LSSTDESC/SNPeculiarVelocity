import pickle
import numpy
import copy
import argparse
import os.path
import array


class HostGalaxies(object):
    """docstring for HostGalaxies"""
    # out dec [0.0161435347419 89.7359068657]
    # out b [-62.7640357412 78.3530163744]

    def __init__(self, sigma_mu=0.08, catseed=1234, seed=123, path='../test/'):
        super(HostGalaxies, self).__init__()
        self.data = pickle.load(open("{}/tenyear.{}.pkl".format(path,catseed), "rb" ))
        self.path = path
        self.galaxies = self.data['galaxies']
        self.catseed=catseed
        self.sigma_mu = sigma_mu
        self.seed=seed

        numpy.random.seed(self.seed)
        rand = []
        for nsn in self.galaxies['nsne']:
            rand.append(numpy.random.normal(scale=sigma_mu,size=nsn))

        self.galaxies['mB'] = []
        for ran , mag in zip(rand, self.galaxies['mB_true_mn']):
            self.galaxies['mB'].append(ran + mag)

        self.xi = None

        if os.path.isfile("{}/pvlist.{}.xi.pkl".format(path,catseed)):
            self.xi = pickle.load(open("{}/pvlist.{}.xi.pkl".format(path,catseed), "rb" ))
        elif os.path.isfile("{}/pvlist.{}.xi".format(path,catseed)):
            ngal = len(self.galaxies['galaxy_id'])
            sz = int((ngal**2+ngal)/2)
            a=array.array('d')
            a.fromfile(open('{}/pvlist.{}.xi'.format(path,catseed),'rb'),sz)
            a= numpy.array(a)
            self.xi = numpy.zeros((ngal,ngal))
            ind = numpy.triu_indices(ngal)
            for v, i, j in zip(a,ind[0],ind[1]):
                self.xi[i,j] =v
                if (i !=j):
                    self.xi[j,i]=v
            pickle.dump(self.xi,open("{}/pvlist.{}.xi.pkl".format(path,catseed), "wb" ))


    def draganFormat(self, sort=False):
        if sort:
            sortin = numpy.argsort(self.galaxies['redshift'])
        else:
            sortin = numpy.arange(len(self.galaxies['redshift']))
        f = open('{}/pvlist.{}.dat'.format(self.path,self.catseed), 'w')
        for i in range(len(sortin)):
            print(' '.join(str(e) for e in (self.galaxies['redshift'][sortin[i]],self.galaxies['mB'][sortin[i]][0],0, \
                self.galaxies['l'][sortin[i]],self.galaxies['b'][sortin[i]], self.galaxies['mB_expected'][sortin[i]])),file=f)
        f.close()

    def getSubset(self,decmin=-90, decmax=90, zmax=0.2, bmin=-90, bmax=90, frac=1):
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
                out['galaxies']["nsne"][i] = found.sum()
                out['galaxies']["mB"][i] = out['galaxies']["mB"][i][found]
                out['galaxies']["random_realize"][i] = out['galaxies']["random_realize"][i][found]                
                sindex += self.galaxies["nsne"][i]

        w = numpy.logical_and.reduce((out['galaxies']['nsne'] > 0,out['galaxies']['dec'] > decmin,
            out['galaxies']['dec'] < decmax, out['galaxies']['b'] >= bmin,
            out['galaxies']['b'] < bmax,
            out['galaxies']['redshift'] < zmax))

        for key, value in out["galaxies"].items():
            if (key != 'random_realize' and key != 'mB'):   #everything is a numpy array
                out["galaxies"][key]=value[w]
            else:               #except this list
                newmB = []
                for ele, use in zip(value, w):
                      if use:
                        newmB.append(ele)
                out["galaxies"][key]=newmB

        if self.xi is None:
            xiout = None
        else:
            xiout = numpy.reshape(self.xi[numpy.outer(w,w)],(w.sum(),w.sum()))

        return out, xiout

    def getIndices(self, x):
        return numpy.searchsorted(self.galaxies["galaxy_id"], x)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma_mu",dest="sigma_mu", default=0.08, type = float, required = False,
                    help="distance modulus standard deviation")
    parser.add_argument("--seed", dest="seed", default=1234, type = int, required = False,
                    help="random number generator seed")
    parser.add_argument('--path', dest='path', default='../test/', type = str, required=False)
    args = parser.parse_args()

    # hg = HostGalaxies(sigma_mu=args.sigma_mu, catseed=args.seed, path=args.path)
    # sg, sxi = hg.getSubset(decmax=60, bmin=-34, bmax=34)
    # print (hg.xi.shape[0], sxi.shape[0])
    HostGalaxies(sigma_mu=args.sigma_mu, catseed=args.seed, path=args.path).draganFormat()