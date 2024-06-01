#!/bin/bash
# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=20:00:00

# set name of job
#SBATCH --job-name=Imagenet_pruning_results

#SBATCH --error=imagenet_pruning_results.err

#SBATCH --output=imagenet_pruning_results.output

# set partition (devel, small, big)

#SBATCH --partition=small

# set number of GPUs
#SBATCH --gres=gpu:1

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=sclaam@leeds.ac.uk
#module load pytorch


which python


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
#    # SAM
#  then
#python Second_order_Receptive_field.py --lr "0.1" --momentum "0.7" --grad_clip "1" --save 1 --experiment 1 --epochs 100 --batch_size 128 --use_scheduler 1 --use_scheduler_batch 0 --num_workers 4 --optimiser "sam" --record 1 -dt $1 --model $2 --RF_level $3 --type $4 --name $5 --save_folder "$HOME/checkpoints"
#  fi
#
python -c "import torch;device = 'cuda' if torch.cuda.is_available() else 'cpu';print(device)"
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
python train_CIFAR10.py  --batch_size 128  --save_folder "/jmain02/home/J2AD014/mtc03/lla98-mtc03/checkpoints" --model $1 --dataset $2 --num_workers $3 --RF_level $4 --type $5 --epochs $6  --name $7 --width $8 --record $9