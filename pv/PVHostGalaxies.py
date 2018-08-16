import numpy
from astropy.coordinates import SkyCoord
import GCRCatalogs
import buzzard_stellar_mass
from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates import SkyCoord
from astropy import units as u
import pickle
import argparse

def main(sigma_MB, seed, test = True):
    # Things that may be user configurable
    controlTime = 1.e1      # number of years

    # The catalog that is used
    if test :
        cat = 'buzzard_v1.6_test'
        path = '../test/'
    else:
        cat = 'buzzard_v1.6'
        path = '../out'
    coadd_cat = GCRCatalogs.load_catalog(cat)

    # Buzzard does not have needed galaxy properties.  The following is used to derive them.
    kfile = "/global/projecta/projectdirs/lsst/groups/CS/Buzzard/metadata/templates/k_nmf_derived.default.fits"
    gdq = buzzard_stellar_mass.GetDerivedQuantities(kfile)

    cosmo = coadd_cat.cosmology

    # Supernova absolute magnitude.  Results don't depend on this since ultimately relative magnitudes used
    MB = -19.25

    # Initialize the seed
    numpy.random.seed(seed)

    # catalog quantities necessary to do the analysis
    quantities = ["redshift", "redshift_true","ra","dec","galaxy_id",
                  "truth/COEFFS/0","truth/COEFFS/1","truth/COEFFS/2","truth/COEFFS/3","truth/COEFFS/4"]

    # GCRCatalog is not efficient for one-step filtering since derived quantities calculated for everything.  Hence do a first coarse filter.

    filts = [
        'redshift < 0.2',
        (numpy.isfinite,"truth/COEFFS/0"),
        (numpy.isfinite,"truth/COEFFS/1"),
        (numpy.isfinite,"truth/COEFFS/2"),
        (numpy.isfinite,"truth/COEFFS/3"),
        (numpy.isfinite,"truth/COEFFS/4"),
        ]

    out = None

    for data in coadd_cat.get_quantities(quantities, filters = filts, return_iterator=True):
        sfr_met_smass=gdq.get_derived_quantities(numpy.array([data["truth/COEFFS/0"],data["truth/COEFFS/1"],
                                                  data["truth/COEFFS/2"],data["truth/COEFFS/3"],
                                                  data["truth/COEFFS/4"]]).T,data["redshift_true"])
        rate = controlTime*(1.23e-03*sfr_met_smass[0]**(.95)+.26e-10*sfr_met_smass[2]**(.72)/(1+data["redshift_true"]))
        nsne = numpy.random.poisson(rate)
        w = nsne > 0
        for key, value in data.items():
            data[key]=value[w]
        data['nsne'] = nsne[w]
        data['mB']  = []
        for nsn, redtrue in zip(data['nsne'], data["redshift_true"]):
            data['mB'].append(MB + numpy.random.normal(scale=sigma_MB,size=nsn)+cosmo.distmod(redtrue).value)
        data['mB'] = numpy.array(data['mB'])
        data['mB_expected'] = MB + cosmo.distmod(data["redshift"]).value
        c = SkyCoord(ra=data["ra"]*u.degree, dec=data["dec"]*u.degree, frame='icrs')
        data['l'] = c.galactic.spherical.lon.value
        data['b'] = c.galactic.spherical.lat.value
        
        if out is None:
            out = data
        else:
            for key, value in data.items():
                out[key] = numpy.append(out[key],value)               


    # Persist results
    ans=dict()
    ans['galaxies']=out
    ans['config']=dict()
    ans['config']['sigma_MB']=sigma_MB
    ans['config']['seed']=seed
    ans['config']['time']=controlTime = 1.e1
    pickle.dump(ans, open( "{}tenyear.{}.{}.pkl".format(path,sigma_MB,seed), "wb" ) )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma_mu",dest="sigma_mu", default=0.08, type = float, required = False,
                    help="distance modulus standard deviation")
    parser.add_argument("--seed", dest="seed", default=1234, type = int, required = False,
                    help="random number generator seed")
    args = parser.parse_args()

    main(args.sigma_mu, args.seed)