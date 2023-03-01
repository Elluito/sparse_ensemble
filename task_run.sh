#!/bin/bash -l
# Submission script for serial Python job
# Luis Alfredo Avendano  2021-10-22

# Run from the current directory and with current environment
#$ -cwd -V

# Ask for some time (hh:mm:ss max of 48:00:00)
#$ -l h_rt=05:00:00

# ASk for some GPU
#$ -l coproc_p100=1

# Ask for some memory (by default, 1G, without a request)
#$ -l h_vmem=8G
# Tell SGE that this is an array job, with "tasks" numbered from 1 to 10
#$ -t 1-10
#  -tc 20
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
#nvcc --version
#~python $1 $2 $3 $4 $5 $6
python main.py -exp 6 -bs 512 --sigma $1 --pruner $2


#&& python main.py && python main.py
