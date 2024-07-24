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
#SBATCH --cpus-per-task=4

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
export LD_LIBRARY_PATH=""
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/jmain02/home/J2AD014/mtc03/lla98-mtc03/.conda/envs/ffcv/lib"
#unset GOMP_CPU_AFFINITY
#unset KMP_AFFINITY
python -c "import os; print(os.environ)"
python -c "import torch;device = 'cuda' if torch.cuda.is_available() else 'cpu';print(device);print('Cuda version with torch: {}'.format(torch.version.cuda))"


#############################################################
#        Create features for given seed name
#############################################################
if [ "${10}" -eq 1 ]; then

python similarity_comparison_architecture.py --ffcv --experiment 1 --architecture $1 --dataset $2 --solution $3 --seedname1 $4 --RF_level $5 --modeltype1 $6 --eval_size $7 --folder $8 --data_folder $9

else

python similarity_comparison_architecture.py --experiment 1 --architecture $1 --dataset $2 --solution $3 --seedname1 $4 --RF_level $5 --modeltype1 $6 --eval_size $7 --folder $8 --data_folder $9

fi
#############################################################
#        Create features for given seed name for logistic regression
#############################################################

#python similarity_comparison_architecture.py --experiment 3 --architecture $1 --solution $2 --seedname1 $3 -rfl $4 --modeltype1 $5 --train $6

#############################################################
#        Train logistic regression on a particular set of features
#############################################################

#python similarity_comparison_architecture.py --experiment 5 --architecture $1 --solution $2 --seedname1 $3 -rfl $4 --modeltype1 $5 --layer_index $6
#############################################################
#       Calculate the similarity of two seeds
#############################################################
#python similarity_comparison_architecture.py --experiment 2 --architecture $1 --seedname1 $2 --seedname2 $3 --modeltype1 $4 --modeltype2 $5 --filetype1 $6 --filetype2 $7

