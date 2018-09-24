# Peculiar Velocity

Determine the precision on f sigma_8 using Type Ia supernova peculiar velocity (magnitude)
autocorrelation.

The parameters we want to explore are:
* Survey duration
* Survey solid angle implemented as narrowing of dec range
* Maximum redshift
* Intrinsic magnitude dispersion

The workflow is as follows:

* Realize supernovae assigned to galaxies in an LSST galaxy catalog.
* Determine the theoretical peculiar magnitude correlation function expected from General Relativity
  * Uses CAMB
  * Populates a GR-based theory matrix *S*.
  * The model for the general theory is *AS*, where the *A* parameter is introduced.  *A*=1 is the GR solution.
* Determine the observed peculiar magnitude for galaxies
  * The magnitude *m* is the average of all supernovae hosted by the galaxy.
  * Assumes that the background distance modulus *mu* is known.
  * Introduces a parameter *M* for the SN Ia absolute magnitude zeropoint.
  * Peculiar magnitude is *m-(mu+M)*.
  * Peculiar magnitudes assigned an uncertainty *sigma/sqrt(n)* where *sigma* is a parameter and *n* is the
number of the SNe in the galaxy.  At this point no measuremnt noise is considered.
* Fit for *A, sigma, M* for different subsets.


## Installation

* Build Dragan's modified NR library.  Instructions on how to build this in `nr/README`.

* Build CAMB.  This is an external submodule located in `pv/CAMB`.  Supposedly new versions of git will give you this.
Otherwise `git submodule update --init --recursive` does the job.  Follow CAMB's build instructions.  Creates an executable
`pv/CAMB/camb`.

* Build Dragan's executable that calculates the theoretical peculiar magnitude correlations. Located in `pv/dragan`.
It uses gsl.  NERSC instructions for cori are
```
module load gsl
module swap PrgEnv-intel PrgEnv-gnu
``` 
Compiled using `make`.  If successful creates the executable `pv/dragan/sig`.

* Python code is dependent on the lsst stack already installed
on cori.  This code is set up using
```
source /global/common/software/lsst/cori-haswell-gcc/stack/setup_current_sims.sh
setup lsst_sims
```

* The executable `camb` is called within `sig`.  It is therefore good to have it in your path.  For example I put
these executables in a `bin` directory.

## Execution

* Make the SNe and galaxies.
```
python PVHostGalaxies.py --no-test
```
* Make ASCII file readable by Dragan's code.  In python:
```
from HostGalaxies import *
HostGalaxies().draganFormat() 
```
* Run Dragan's code
```
srun -n  5248 -c 1 --cpu_bind=cores ./sig 0 $HOME/desc/PeculiarVelocity/out/pv.1234
```
* Fit parameters.
```
python fit.py
```
* Look at output.
```
python view.py
```
## Notes for Brian

The code that we would like to make faster is sig.  It reads the file pvlist.1234.dat.  There are two versions of this file.  In /global/homes/a/akim/desc/PeculiarVelocity/out there is the real one we want to use.  In /global/homes/a/akim/desc/PeculiarVelocity/test/ there is a short one that completes quickly.

Another code that is slow is fit.py, where there is a matrix inversion and calculation of eigenvalues of a large matrix.  The calculations are done using numpy, for which cori knows to run openmp maybe.

started at 5,643 
