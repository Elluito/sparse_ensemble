#!/bin/bash
# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=12:00:00

# set name of job
#SBATCH --job-name=zero_cost_nas_rf

#SBATCH --error=zero_cost_nas_rf.err

#SBATCH --output=zero_cost_nas_rf.output

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
which python

#python train_CIFAR10.py --model $1 --dataset $2 --num_workers $3 --RF_level $4 --type $5

#python Second_order_Receptive_field.py --experiment 2 --optimiser "kfac"
#
#python Second_order_Receptive_field.py --experiment 2 --optimiser "sam"
export LD_LIBRARY_PATH=""
l=$(which python)
lib_path_of_current_enviroment="${l%%python}"
echo "Ld library ${lib_path_of_current_enviroment}"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$lib_path_of_current_enviroment
export PYTHONPATH="/jmain02/home/J2AD014/mtc03/lla98-mtc03/.conda/envs/ffcv/lib/python3.9/site-packages"


#  if [ $6 -eq 1 ]
#  then
#    # KFac
#python3.9 Second_order_Receptive_field.py --lr "0.01" --momentum "0.5" --grad_clip $6 --save 1 --experiment 1 --record_time 1 --epochs 100 --batch_size 128 --use_scheduler 1 --use_scheduler_batch 0 --num_workers 4 --optimiser "ekfac" --record 1 -dt $1 --model $2 --RF_level $3 --type $4 --name $5 --save_folder "$HOME/checkpoints"
#
#  fi
#
#  if [ $6 -eq 2 ]
#    # SAM
#  then
#python3.9 Second_order_Receptive_field.py --lr "0.1" --momentum "0.7" --grad_clip "1" --save 1 --experiment 1 --record_time 1 --record_flops 1 --record_saturation 1 --epochs 200 --batch_size 128 --use_scheduler 1 --use_scheduler_batch 0 --num_workers 4 --optimiser "sam" --record 1 -dt $1 --model $2 --RF_level $3 --type $4 --name $5 --save_folder "$HOME/second_order_experiments"
#  fi

#python3.9 Second_order_Receptive_field.py --level $1
python zero_cost_nas_RF.py