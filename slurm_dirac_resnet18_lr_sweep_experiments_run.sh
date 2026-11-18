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

#SBATCH --partition=gpu

# set number of GPUs
#SBATCH --gres=gpu:1

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4GB

# send mail to this address
#SBATCH --mail-user=l.avendanomunoz@leeds.ac.uk

export LD_LIBRARY_PATH=""
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/users/$USER/.conda/envs/work/lib"
export PYTHONPATH="/users/$USER/.conda/envs/work/lib/python3.9/site-packages"

#############################################################
#     SGD training run for the DiracNet-50 / ResNet-18 learning-rate sweep, no FFCV.
#     $1  model             (diracnet50, resnet18)
#     $2  dataset           (cifar10)
#     $3  num_workers
#     $4  RF_level
#     $5  epochs
#     $6  name              (run identifier, has the lr baked into it)
#     $7  lr
#     $8  input_resolution
#     $9  save_folder
#
#     diracnet50 uses the DiracNet parameter groups (no weight decay on 1-D params) inside train_CIFAR10.py;
#     --dirac_width 4 must match the value used for pruning. No --record_flops: the controls were trained
#     without it and FLOPs are not needed for the mediation analysis.
#############################################################

python train_CIFAR10.py --experiment 1 --model "$1" --dataset "$2" --num_workers "$3" \
  --RF_level "$4" --type "normal" --epochs "$5" --name "$6" --lr "$7" --input_resolution "$8" \
  --batch_size 128 --save_folder "$9" --data_folder "${SCRATCH}/data2" \
  --width 1 --record 1 --record_saturation 1 --dirac_width 4
