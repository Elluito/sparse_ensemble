#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=00:09:00

# set name of job
#SBATCH --job-name=pytorch_test

#SBATCH --error=pytorch_test.err

#SBATCH --output=pytorch_test.output

# set partition (devel, small, big)

#SBATCH --partition=small

# set number of GPUs
#SBATCH --gres=gpu:1

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4GB

# send mail to this address
#SBATCH --mail-user=sclaam@leeds.ac.uk

export LD_LIBRARY_PATH=""
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/users/sclaam/.conda/envs/work/lib"
export PYTHONPATH="/users/sclaam/.conda/envs/work/lib/python3.9/site-packages"

#############################################################
#     Train  with out FFCV
#############################################################


python train_CIFAR10.py --batch_size 128  --save_folder "${SCRATCH}/diverse_pooling_100" --model $1 --dataset $2 --num_workers $3 --RF_level $4 --type $5 --epochs $6  --name $7 --width $8 --record $9 --input_resolution "${10}"
