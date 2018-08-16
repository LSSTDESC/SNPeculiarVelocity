# Peculiar Velocity

Determine the precision on $f\sigma_8$.

## Installation

There are several distinct codes.

nr/ : c code.  Instructions on how to build this in nr/README

pv/ : python code.  It is dependent on the lsst stack already installed
on cori.  This code is set up using

source /global/common/software/lsst/cori-haswell-gcc/stack/setup_current_sims.sh
setup lsst_sims

pv/dragan/ : c code.  Uses gsl.  According to cori instructions the following commands must be done

module load gsl
module swap PrgEnv-intel PrgEnv-gnu

compiled using make.  If successful creates the executable pv/dragan/sig

pv/CAMB/ : fortran code.  It is an external modules. it is build using make. If successful creates the executable pv/CAMB/sig.  This executable is called by sig so needs to be in the PATH.



## Execution

1) Make the SNe and galaxies
> python PVHostGalaxies.py 

2) Make ASCII file readable by Dragan's code
In [1]: from HostGalaxies import *
In [2]: HostGalaxies().draganFormat() 

3) Run Dragan's code from directory where ascii file is created
>sig

4) Fit parameters
> python fit.py

5) Look at output
> python view.py

## Notes for Brian

The code that we would like to make faster is sig.  It reads the file pvlist.1234.dat.  There are two versions of this file.  In /global/homes/a/akim/desc/PeculiarVelocity/out there is the real one we want to use.  In /global/homes/a/akim/desc/PeculiarVelocity/test/ there is a short one that completes quickly.
