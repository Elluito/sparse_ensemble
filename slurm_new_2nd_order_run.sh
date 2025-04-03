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
#module load pytorch

#which python

#python train_CIFAR10.py --model $1 --dataset $2 --num_workers $3 --RF_level $4 --type $5

#python Second_order_Receptive_field.py --experiment 2 --optimiser "kfac"
#
#python Second_order_Receptive_field.py --experiment 2 --optimiser "sam"

#  if [ $6 -eq 1 ]
#  then
#    # KFac
#python Second_order_Receptive_field.py --lr "0.01" --momentum "0.5" --grad_clip "1" --save 1 --experiment 1 --epochs 100 --batch_size 32 --use_scheduler 1 --use_scheduler_batch 0 --num_workers 4 --optimiser "kfac" --record 1 -dt $1 --model $2 --RF_level $3 --type $4 --name $5 --save_folder "$HOME/checkpoints"
#
#  fi
#
#  if [ $6 -eq 2 ]
#    # ASAM
#  then
#python Second_order_Receptive_field.py --lr "0.1" --momentum "0.7" --grad_clip "1" --save 1 --experiment 1 --epochs 100 --batch_size 128 --use_scheduler 1 --use_scheduler_batch 0 --num_workers 4 --optimiser "sam" --record 1 -dt $1 --model $2 --RF_level $3 --type $4 --name $5 --save_folder "$HOME/checkpoints"
#  fi
#
#python -c "import torch;device = 'cuda' if torch.cuda.is_available() else 'cpu';print(device);print('Cuda version with torch: {}'.format(torch.version.cuda))"
#python -c "import os; print(os.environ)"
#printf "Start Test \n"
#python test_backwards.py
#printf "End Test \n"
#module load pytorch
#echo "After loading the pytorch module"
#which python
#python -c "import torch;device = 'cuda' if torch.cuda.is_available() else 'cpu';print(device);print('Cuda version with torch: {}'.format(torch.version.cuda))"
#python -c "import os; print(os.environ)"
#printf "Start Test \n"
#python test_backwards.py
#printf "End Test \n"
##echo "============ 2 workers ============================"
##python hao_models_pruning_test.py --workers 2
#echo "============ 4 workers ============================"
#python hao_models_pruning_test.py --workers 4
#echo "============ 8 workers ============================"
#python hao_models_pruning_test.py --workers 2 --experiment 2 --model $1
#echo "============ 16 workers ============================"
#python hao_models_pruning_test.py --workers 16
#echo "============ 32 workers ============================"
#python hao_models_pruning_test.py --workers 32

#export LD_LIBRARY_PATH=""
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/jmain02/home/J2AD014/mtc03/lla98-mtc03/.conda/envs/ffcv/lib"


#unset GOMP_CPU_AFFINITY
#unset KMP_AFFINITY

#python -c "import torch;device = 'cuda' if torch.cuda.is_available() else 'cpu';print(device);print('Cuda version with torch: {}'.format(torch.version.cuda))"

export LD_LIBRARY_PATH=""
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/users/sclaam/.conda/envs/work/lib"
export PYTHONPATH="/users/sclaam/.conda/envs/work/lib/python3.9/site-packages"

#export LD_LIBRARY_PATH=""
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/jmain02/home/J2AD014/mtc03/lla98-mtc03/.conda/envs/ffcv/lib"
#export PYTHONPATH="/jmain02/home/J2AD014/mtc03/lla98-mtc03/.conda/envs/ffcv/lib/python3.9/site-packages"


  if [ $6 == "ekfac" ]
  then

    # Kfac

#python3.9 Second_order_Receptive_field.py --lr "0.01" --momentum "0.5" --grad_clip $6 --save 1 --experiment 1 --record_time 1 --record_flops 1 --record_saturation 1 --epochs 200 --batch_size 128 --use_scheduler 1 --use_scheduler_batch 0 --num_workers 4 --optimiser "ekfac" --record 1 -dt $1 --model $2 --RF_level $3 --type $4 --name $5 --save_folder "$HOME/second_order_experiments"

python Second_order_Receptive_field.py --lr "0.1" --momentum "0.9"  --save 1 --experiment 1 --batch_size 128 --use_scheduler 1 --use_scheduler_batch 0 --num_workers 8 --optimiser "ekfac" --record 1 --save_folder "$SCRATCH/second_order_checkpoints" --dataset $1 --model $2 --RF_level $3 --type $4 --name $5 --grad_clip $7 --record_saturation $8  --epochs $9 --input_resolution "${10}"

  fi
  if [ $6 == "sam" ]

    # ASAM
  then

#python3.9 Second_order_Receptive_field.py --lr "0.1" --momentum "0.9" --grad_clip $6 --save 1 --experiment 1 --record_time 1 --record_flops 1 --record_saturation 1 --epochs 100 --batch_size 128 --use_scheduler 1 --use_scheduler_batch 0 --num_workers 0 --optimiser "sam" --record 1 -dt $1 --model $2 --RF_level $3 --type $4 --name $5 --save_folder "$HOME/second_order_experiments"

python Second_order_Receptive_field.py --lr "0.01" --momentum "0.9" --batch_size 128 --use_scheduler 1 --use_scheduler_batch 0 --num_workers 8  --save 1 --experiment 1 --optimiser "sam" --record 1 --save_folder "$SCRATCH/second_order_checkpoints"  --dataset $1 --model $2 --RF_level $3 --type $4 --name $5 --grad_clip $7  --record_saturation $8 --epochs $9 --input_resolution "${10}"

  fi

#python3.9 Second_order_Receptive_field.py --level $1
#python zero_cost_nas_RF.py
