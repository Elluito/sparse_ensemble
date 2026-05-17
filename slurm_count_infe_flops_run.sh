#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=00:15:00

# set name of job
#SBATCH --job-name=infe_flops

#SBATCH --error=infe_flops.err

#SBATCH --output=infe_flops.output

# set partition
#SBATCH --partition=small

# set number of GPUs
#SBATCH --gres=gpu:1

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4GB

# send mail to this address
#SBATCH --mail-user=sclaam@leeds.ac.uk

export LD_LIBRARY_PATH=""
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/users/sclaam/.conda/envs/work/lib"
export PYTHONPATH="/users/sclaam/.conda/envs/work/lib/python3.9/site-packages"

# Arguments:
#  $1  model            (e.g. resnet18, resnet50, vgg19)
#  $2  dataset          (e.g. cifar10, cifar100, tiny_imagenet)
#  $3  num_workers
#  $4  RF_level
#  $5  type             (e.g. normal, pytorch)
#  $6  name             (unique identifier string)
#  $7  width
#  $8  input_resolution
#  $9  resume_solution  (full path to .pth checkpoint)
# $10  save_folder      (directory where the output CSV is written)

python train_CIFAR10.py \
    --experiment 4 \
    --model $1 \
    --dataset $2 \
    --num_workers $3 \
    --RF_level $4 \
    --type $5 \
    --name $6 \
    --width $7 \
    --input_resolution $8 \
    --resume_solution $9 \
    --save_folder "${10}" \
    --pruning_type "${11}" \
    --batch_size 128
