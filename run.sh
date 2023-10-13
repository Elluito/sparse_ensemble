#!/bin/bash -l
# Submission script for serial Python job
# Luis Alfredo Avendano  2021-10-22

# Run from the current directory and with current environment
#$ -cwd -V

# Ask for some time (hh:mm:ss max of 48:00:00)
#$ -l h_rt=5:00:00


# ASk for some GPU
#$ -l coproc_p100=1

# Ask for some memory (by default, 1G, without a request)
#$ -l h_vmem=50G
# -t 1-2
# -pe smp 3
# Send emails when job starts and ends
#$ -m be
#export OMP_NUM_THREADS=3
#export MKL_NUM_THREADS=3
#export OPENBLAS_NUM_THREADS=1
# Now run the job 
#module load intel openmpi
#module add anaconda
#module add cuda/11.1.1
conda activate work2
#which python
unset GOMP_CPU_AFFINITY
unset KMP_AFFINITY
#nvcc --version
#python main.py $1 $2 $3 $4 $5 $6
#&& python main.py && python main.py
#
#CUDA_VISIBLE_DEVICES="0" accelerate launch --num_processes=1 --mixed_precision=fp16 main.py -exp $1 -bs 128 --sigma $2 --pruner $3 --architecture $4 --dataset $5 --pruning_rate $6 --modeltype $7 --epochs $8
#python main.py -exp $1 -bs 128 --sigma $2 --pruner $3 --architecture $4 --dataset $5 --pruning_rate $6 --modeltype $7 --num_workers 10 --epochs $8 -pop 1 --one_batch False -sa $9 -ls True -tr 600 --functions "${10}"
#python change_files.py
#python main.py -exp $1 -bs 128 --sigma $2 --pruner $3 --architecture $4 --dataset $5 --pruning_rate $6 --modeltype $7 --num_workers 10 --epochs $8 -pop 1 --one_batch False -sa nsg -ls True -tr 300 --functions 2
#python prune_models.py
#python smoothness_measurement.py --model $1 --dataset $2 --RF_level $3 --type $4 --name $5
#python train_CIFAR10.py --model $1 --dataset $2 --num_workers $3 --RF_level $4 --type $5
#python main.py -exp $1 -bs 128 --sigma $2 --pruner $3 --architecture $4 --dataset $5 --pruning_rate $6 --modeltype $7 --num_workers 10 --epochs $8 -pop 1 --one_batch False -sa tpe -ls True -tr 300 --functions 1
#python main.py -exp $1 -bs 128 --sigma $2 --pruner $3 --architecture $4 --dataset $5 --pruning_rate $6 --modeltype $7 --num_workers 10 --epochs $8 -pop 1 --one_batch False -sa tpe -ls True -tr 300 --functions 2


# To save the representations
python similarity_comparison_architecture.py --architecture $1 --seedname2 $2 --seedname1 $3
