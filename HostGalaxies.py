import pickle

class HostGalaxies(object):
    """docstring for HostGalaxies"""

    def __init__(self, sigma_mu=0.08, seed=1234):
        super(HostGalaxies, self).__init__()
        self.galaxies = pickle.load("tenyear.{}.{}.pkl".format(sigma_mu,seed), "rb" )

    def draganFormat(self):
        sortin = numpy.argsort(self.galaxies['redshift'])
        f = open('pvlist.{}.dat'.format(seed), 'w')
        for i in range(len(sortin)):
            print(' '.join(str(e) for e in (self.galaxies['redshift'][sortin[i]],self.galaxies['mB'][sortin[i]],0, \
                self.galaxies['l'][sortin[i]],self.galaxies['b'][sortin[i]], self.galaxies['mB_expected'][sortin[i]], \
                self.galaxies['nsne'][sortin[i]])),file=f)
            f.close()


foo = HostGalaxies()
foo.draganFormat()