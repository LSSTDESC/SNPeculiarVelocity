#!/bin/bash
#SBATCH -N 80
#SBATCH -C haswell
#SBATCH -q regular
#SBATCH -J ximaker
#SBATCH --mail-user=agkim@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -t 01:00:00
#SBATCH -A dessn

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#cd $HOME/desc/PeculiarVelocity/bin

export PATH="/global/homes/a/akim/desc/PeculiarVelocity/bin:$PATH"

#run the application on 80 nodes:
srun -n 5120 -c 1 --cpu_bind=cores sig $HOME/desc/PeculiarVelocity/out/pvlist.1234 0
#srun -n  5248 -c 1 --cpu_bind=cores ./sig 0 $HOME/desc/PeculiarVelocity/out/pv.1234
