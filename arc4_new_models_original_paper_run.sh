#!/bin/bash
#$ -cwd -V
# -l h_rt=01:00:00
#$ -pe smp 8
# Send emails when job starts and ends
#$ -m be
#$ -l h_vmem=16G
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
export LD_LIBRARY_PATH=""
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/nobackup/sclaam/.conda/envs/work/lib"
#export PYTHONPATH="/jmain02/home/J2AD014/mtc03/lla98-mtc03/.conda/envs/ffcv/lib/python3.9/site-packages"
conda activate work
unset GOMP_CPU_AFFINITY KMP_AFFINITY

#############################################################
#     Train  with FFCV
#############################################################


python train_CIFAR10.py --record_time --record_flops --batch_size 128  --save_folder "/nobackup/sclaam/models_original_RF_paper" --model $1 --dataset $2 --num_workers $3 --RF_level $4 --type $5 --epochs $6  --name $7 --width $8 --record $9
