#!/bin/bash
# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=50:00:00

# set name of job
#SBATCH --job-name=Optuna_tuning

# set partition (devel, small, big)
#SBATCH --partition=small

# set number of GPUs
#SBATCH --gres=gpu:1

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=sclaam@leeds.ac.uk

conda active work2

python train_CIFAR10.py --model $1 --dataset $2 --num_workers $3 --RF_level $4 --type $5

