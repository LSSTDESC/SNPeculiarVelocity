#!/bin/bash -l
#SBATCH -N 80
#SBATCH -C haswell
#SBATCH -q regular
#SBATCH -J ximaker
#SBATCH --mail-user=cju@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -t 00:10:00

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#cd $HOME/PeculiarVelocity/bin

export PATH="/project/projectdirs/m1727/akim/pvoutcosmo/"

#run the application on 80 nodes:
srun -n 5120 -c 1 --cpu_bind=cores ./sig /project/projectdirs/m1727/akim/pvoutcosmo/pvlist.1234 0
#srun -n  5248 -c 1 --cpu_bind=cores ./sig 0 /project/projectdirs/m1727/akim/pvoutcosmo/pvlist.1234
