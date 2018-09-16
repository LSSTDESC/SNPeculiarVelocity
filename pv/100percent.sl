#!/bin/bash
#SBATCH -C haswell
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -J 100percent
#SBATCH --mail-user=agkim@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -t 48:00:00
#SBATCH -A dessn

module load python
source activate py36
cd $HOME/desc/PeculiarVelocity/pv/

export OMP_NUM_THREADS=64
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
#run the application:
srun -n 1 -c 64 --cpu-bind=sockets  python fit.py --path ../out/ --nchain 2000 --savef 20
