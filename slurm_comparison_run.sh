#!/bin/bash
# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=00:30:00

# set name of job
#SBATCH --job-name=compare_small_imagenet

#SBATCH --error=compare_small_imagenet.err

#SBATCH --output=compare_small_imagenet.output

# set partition (devel, small, big)

#SBATCH --partition=small

# set number of GPUs
#SBATCH --gres=gpu:1

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=sclaam@leeds.ac.uk
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
python compare_small_imagenet.py