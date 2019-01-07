#!/bin/bash
#SBATCH -C haswell
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -J 100percent
#SBATCH --mail-user=cju@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -t 20:00:00

module load python/3.6-anaconda-4.4
source activate py36
cd $HOME/PeculiarVelocity/pv/

export OMP_NUM_THREADS=64
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
#run the application:
srun -n 1 -c 64 --cpu-bind=sockets  python fit.py --sigma_mu 0.08 --path /project/projectdirs/m1727/akim/pvoutcosmo --frac 0.65 --nchain 2000 --savef 20 --zmax 0.07
