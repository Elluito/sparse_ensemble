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
export LD_LIBRARY_PATH=""
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/jmain02/home/J2AD014/mtc03/lla98-mtc03/.conda/envs/ffcv/lib"
#unset GOMP_CPU_AFFINITY
#unset KMP_AFFINITY
python -c "import os; print(os.environ)"
python -c "import torch;device = 'cuda' if torch.cuda.is_available() else 'cpu';print(device);print('Cuda version with torch: {}'.format(torch.version.cuda))"

#############################################################
#     Hessian calculation with FFCV
#############################################################


python train_CIFAR10.py --ffcv --record_time --record_flops --batch_size 128  --save_folder "/jmain02/home/J2AD014/mtc03/lla98-mtc03/checkpoints" --model $1 --dataset $2 --num_workers $3 --RF_level $4 --type $5 --epochs $6  --name $7 --width $8 --record $9  --ffcv_train "${10}" --ffcv_val "${11}"

#############################################################
#     Train iterative RF with FFCV
#############################################################

#python train_CIFAR10.py  --experiment 2 --batch_size 128 --ffcv --record_time --record_flops --save_folder "/jmain02/home/J2AD014/mtc03/lla98-mtc03/checkpoints" --model $1 --dataset $2 --num_workers $3 --RF_level $4 --type $5 --epochs $6  --name $7 --width $8 --record $9  --ffcv_train "${10}" --ffcv_val "${11}"
#model="resnet50"
#dataset="cifar10"
#init=true
#if [ $init ]; then
#    solution_string="initial_weights"
#
#else
#
#
#    solution_string="test_acc"
#fi
#
#
#directory=/nobackup/sclaam/checkpoints
#resnet50_normal_cifar10_seed_0_3_rf_level_1_initial_weights.pth
#level_1_seed0=($(ls $directory | grep -i "${model}.*${dataset}.*_seed_0_.*_level_1_${solution_string}.*"))
#echo $level_1_seed0
##level_1_seed1=($(ls $directory | grep -i "vgg19.*_seed_1_rf_level_1_test_acc.*"))
##echo $level_1_seed1
##
##level1_seeds=($level_1_seed0 $level_1_seed1)
##
##level_2_seed0=($(ls $directory | grep -i "vgg19.*_seed_0_rf_level_2_test_acc.*"))
#level_2_seed0=($(ls $directory | grep -i "${model}.*${dataset}.*_seed_0_.*_level_2_${solution_string}.*"))
#echo $level_2_seed0
##level_2_seed1=($(ls $directory | grep -i "vgg19.*_seed_1_rf_level_2_test_acc.*"))
##echo $level_2_seed1
##
##level2_seeds=($level_2_seed0 $level_2_seed1)
##
#level_3_seed0=($(ls $directory | grep -i "${model}.*${dataset}.*_seed_0_.*_level_3_${solution_string}.*"))
##level_3_seed0=($(ls $directory | grep -i "vgg19.*_seed_0_rf_level_3_test_acc.*"))
##
#echo $level_3_seed0
##level_3_seed1=($(ls $directory | grep -i "vgg19.*_seed_1_rf_level_3_test_acc.*"))
##echo $level_3_seed1
##
##level3_seeds=($level_3_seed0 $level_3_seed1)
##
#
##level_4_seed0=($(ls $directory | grep -i "vgg19.*_seed_0_rf_level_4_test_acc.*"))
#level_4_seed0=($(ls $directory | grep -i "${model}.*${dataset}.*_seed_0_.*_rf_level_4_${solution_string}.*"))
#echo $level_4_seed0
##level_4_seed1=($(ls $directory | grep -i "vgg19.*_seed_1_rf_level_4_test_acc.*"))
##echo $level_4_seed1
##level4_seeds=($level_4_seed0 $level_4_seed1)
#
#
#seeds="0"
#rf_levels=(1 2 3 4)
#levels_max=${#rf_levels[@]}                                  # Take the length of that array
#for ((idxA=0; idxA<levels_max; idxA++)); do              # iterate idxA from 0 to length
#
#
##levels_by_seed=(${level1_seeds[$idxB]} ${level2_seeds[$idxB]} ${level3_seeds[$idxB]} ${level4_seeds[$idxB]})
#
#levels_by_seed=(${level_1_seed0} ${level_2_seed0} ${level_3_seed0} ${level_4_seed0})
#echo "${directory}/${levels_by_seed[$idxA]}"
##echo "${levels_by_seed}"
#
#qsub -N "${model}_hessian_init_${dataset}_${rf_levels[$idxA]}" run.sh  "${model}" "${dataset}" "${rf_levels[$idxA]}" "normal" "seed_0_rf_level_${rf_levels[$idxA]}_init" "${directory}/${levels_by_seed[$idxA]}"
#
#done
