from astropy.cosmology import FlatLambdaCDM
import fitsio
import numpy as np
import sys

class GetDerivedQuantities:
    
    def sfr(self, coeff, z):
 
        #get angular diameter distances

        da    = self.cosmo.angular_diameter_distance(z).value  
        ssfr  = np.dot(coeff, self.sfh_tot)
        smass = np.dot(self.mass_tot, coeff.T) * (da * 1e6 / 10) ** 2
        sfr   = ssfr * smass[:,np.newaxis]

        #get the values at z_galaxy
        sfr   = sfr[:,-1]

        return sfr
    
    def met(self, coeff, z):
 
        #get angular diameter distances

        da    = self.cosmo.angular_diameter_distance(z).value  
        ssfr  = np.dot(coeff, self.sfh_tot)
        met   = np.dot(coeff, self.sfh_tot * self.sfh_met) / ssfr
   
        #get the values at z_galaxy
        met   = met[:,-1]

        return met
    
    def smass(self, coeff, z):
 
        #get angular diameter distances

        da    = self.cosmo.angular_diameter_distance(z).value  
        smass = np.dot(self.mass_tot, coeff.T) * (da * 1e6 / 10) ** 2

        return smass
    
    def get_derived_quantities(self,coeff, z): 
        """
        Given a path to a kcorrect template file, as well as 
        kcorrect coefficients for a set of galaxies and return derived
        quantities

        inputs
        ------
        template_path -- str
        path to kcorrect templates
        coeff -- (N x N_templates) array
        array of kcorrect coefficients
        z -- (N,) array
        redshifts of galaxies

        returns
        -------
        sfr -- (N,) array
        Array of star formation rates for galaxies
        met -- (N,) array
        Array of metallicities for galaxies
        smass -- (N,) array
        Array of stellar masses for galaxies
        """


        #get angular diameter distances
 
        da    = self.cosmo.angular_diameter_distance(z).value  

        smass = np.dot(self.mass_tot, coeff.T) * (da * 1e6 / 10) ** 2
        ssfr  = np.dot(coeff, self.sfh_tot)
        # met   = np.dot(coeff, self.sfh_tot * self.sfh_met) / ssfr
        sfr   = ssfr * smass[:,np.newaxis]

        #get the values at z_galaxy
        # met   = met[:,-1]
        sfr   = sfr[:,-1]

        return np.vstack((sfr, np.zeros(len(smass)), smass))

    def __init__(self,template_path):
        self.sfh_tot = fitsio.read(template_path, 12)
        self.sfh_met = fitsio.read(template_path, 13)
        self.mass_tot = fitsio.read(template_path, 17)
        self.cosmo = FlatLambdaCDM(100, 0.286)
        
def get_derived_quantities(template_path, coeff, z): 
  """
  Given a path to a kcorrect template file, as well as 
  kcorrect coefficients for a set of galaxies and return derived
  quantities
  
  inputs
  ------
  template_path -- str
    path to kcorrect templates
  coeff -- (N x N_templates) array
    array of kcorrect coefficients
  z -- (N,) array
    redshifts of galaxies
    
  returns
  -------
  sfr -- (N,) array
    Array of star formation rates for galaxies
  met -- (N,) array
    Array of metallicities for galaxies
  smass -- (N,) array
    Array of stellar masses for galaxies
  """
  
  #read in relevant template info
  sfh_tot = fitsio.read(template_path, 12)
  sfh_met = fitsio.read(template_path, 13)
  mass_tot = fitsio.read(template_path, 17)
  
  #get angular diameter distances
  cosmo = FlatLambdaCDM(100, 0.286)
  da    = cosmo.angular_diameter_distance(z).value  
  
  smass = np.dot(mass_tot, coeff.T) * (da * 1e6 / 10) ** 2
  ssfr  = np.dot(coeff, sfh_tot)
  met   = np.dot(coeff, sfh_tot * sfh_met) / ssfr
  sfr   = ssfr * smass[:,np.newaxis]
  
  #get the values at z_galaxy
  met   = met[:,-1]
  sfr   = sfr[:,-1]
  
  return numpy.vstack((sfr, met, smass))
  

if __name__=='__main__':
    kfile = sys.argv[1] #name of file containing 
    filename = sys.argv[2] #name of galaxy catalog file

    galaxies  = fitsio.FITS(filename)[-1].read(columns=['COEFFS','Z','DELTAM']) # read relevant info from files
    coeffs    = galaxies['COEFFS']*10**(galaxies['DELTAM'].reshape(-1,1)/2.5)
    sfr, met, smass = get_derived_quantities(kfile,coeffs,galaxies['Z'])