#!/bin/bash -l
# Submission script for serial Python job

# Luis Alfredo Avendano  2021-10-22

# Run from the current directory and with current environment
#$ -cwd -V

# Ask for some time (hh:mm:ss max of 48:00:00)
#$ -l h_rt=08:00:00

# ASk for some GPU
#$ -l coproc_p100=1

# Ask for some memory (by default, 1G, without a request)
#$ -l h_vmem=8G
# Send emails when job starts and ends
#$ -m e

module add anaconda
module add cuda/11.1.1
conda activate work
python test_imagenet.py
