#!/bin/bash -l
# Submission script for serial Python job
# Luis Alfredo Avendano  2021-10-22

# Run from the current directory and with current environment
#$ -cwd -V

# Ask for some time (hh:mm:ss max of 48:00:00)
#$ -l h_rt=40:00:00

# ASk for some GPU
#$ -l coproc_k80=1

# Ask for some memory (by default, 1G, without a request)
#$ -l h_vmem=128G
# -pe smp 3
# Send emails when job starts and ends
#$ -m e
#export OMP_NUM_THREADS=3
#export MKL_NUM_THREADS=3
#export OPENBLAS_NUM_THREADS=1
# Now run the job
#module load intel openmpi
module add anaconda
module add cuda/11.1.1
conda activate work
which python
unset GOMP_CPU_AFFINITY
#nvcc --version
#~python $1 $2 $3 $4 $5 $6
CUDA_LAUNCH_BLOCKING=1 python main.py -exp $1 -bs 4 --sigma $2 --pruner $3 --architecture $4 --dataset $5 --pruning_rate $6 --modeltype $7 --epochs $8 --num_workers 4
 #&& python main.py && python main.py
