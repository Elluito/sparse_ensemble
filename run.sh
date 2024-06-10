#!/bin/bash -l
# Submission script for serial Python job
# Luis Alfredo Avendano  2021-10-22

# Run from the current directory and with current environment
#$ -V -cwd

# Ask for some time (hh:mm:ss max of 00:10:00)
#$ -l h_rt=07:00:00

# ASk for some GPU
# -l coproc_p100=1

# Ask for some memory (by default, 1G, without a request)
# -l node_type=40core-768G
# -l h_vmem=600G
#$ -l h_vmem=120G
# -t 2-5
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
conda activate work
which python
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
#python train_CIFAR10.py --model $1 --dataset $2 --num_workers $3 --RF_level $4 --type $5
#python main.py -exp $1 -bs 128 --sigma $2 --pruner $3 --architecture $4 --dataset $5 --pruning_rate $6 --modeltype $7 --num_workers 10 --epochs $8 -pop 1 --one_batch False -sa tpe -ls True -tr 300 --functions 1
#python main.py -exp $1 -bs 128 --sigma $2 --pruner $3 --architecture $4 --dataset $5 --pruning_rate $6 --modeltype $7 --num_workers 10 --epochs $8 -pop 1 --one_batch False -sa tpe -ls True -tr 300 --functions 2


# To save the representations
#echo "First similarity function"
########################################################
#                Change type of file
########################################################
#python change_files.py --architecture $1 --modeltype hub --seedname1 $2


#############################################################
#        Create features for given seed name
#############################################################

#python similarity_comparison_architecture.py --experiment 1 --architecture $1 --solution $2 --seedname1 $3 -rfl $4 --modeltype1 $5

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

#############################################################
#       Calculate the describing statistics of all feature layers of a solution
#############################################################
#
#python similarity_comparison_architecture.py --experiment 4 --architecture $1 --seedname1 $2 --seedname2 $3 --modeltype1 $4 --modeltype2 $5 --filetype1 $6 --filetype2 $7

#############################################################
#     Smoothness (Hessian) calculation
#############################################################

#python smoothness_measurement.py --model $1 --dataset $2 --RF_level $3 --type $4 --name $5 --solution $6

#############################################################
#     Training a model with specific RF
#############################################################
#
#python train_CIFAR10.py --model $1 --dataset $2 --num_workers $3 --RF_level $4 --type $5 --epochs $6  --name $7 --width $8 --record $9 --batch_size 128 --save_folder "${10}"

#############################################################
#     Resume Training of a model with specific RF from solution
#############################################################


#python train_CIFAR10.py --model $1 --dataset $2 --num_workers $3 --RF_level $4 --type $5 --epochs $6  --name $7 --width $8 --record $9 --batch_size 128 --save_folder "${10}" --

#############################################################
#     One shot pruning results
#############################################################
#python prune_models.py  --model $1 --dataset $2 --num_workers $3 --RF_level $4 --type $5 --folder $6 --experiment $7
#--name $8
#--name $7

#############################################################
#     Fine tuning pruning results
#############################################################
#python prune_models.py  --model $1 --dataset $2 --num_workers $3 --RF_level $4 --type $5 --folder $6 --name $7 --solution $8 --pruning_rate "${9}" --experiment "${10}"
#python prune_models.py  --model $1 --dataset $2 --num_workers $3 --RF_level $4 --type $5 --folder $6 --solution $7 --experiment $8

#############################################################
#     One shot with specific pruning rate results
#############################################################

#CONDA_BASE_DIR=~/home/luisaam/anaconda3
#
#cd "$1"
#
## Activate the conda environment.
#source "$CONDA_BASE_DIR/etc/profile.d/conda.sh"
#conda activate "$0"
#eval "$(conda shell.bash hook)"
#conda activate work
#which python

#python prune_models.py  --model $1 --dataset $2 --num_workers $3 --RF_level $4 --type $5 --folder $6 --pruning_rate $7 --experiment $8

#echo  $1  $2  $3  $4  $5 $6  $7  $8  $9 "${10}"

#python prune_models.py  --model "vgg19" --dataset "tiny_imagenet" --num_workers "4" --RF_level 4 --type "normal" --folder "~/" --solution "vgg19_normal_tiny_imagenet_seed_3_rf_level_4_test_acc_39.03.pth" --pruning_rate 0.9 --experiment 2



#################################################################
#             Stochastic pruning
#################################################################

#############################################################
#     Fine tuning pruning results
#############################################################
#python main.py -exp $1 -bs 128 --sigma $2 --pruner $3 --architecture $4 --dataset $5 --pruning_rate $6 --modeltype $7 --epochs $8 -pop 10


#############################################################
#     Linear Intrepolation experiments
#############################################################
#
#type="one_shot"
#  if [ $1 -eq 0 ]
#  then
#    python stochastic_loss_landscape.py --sigma "0.005" --batch_size 128 --pruner "global" -pru "0.9" -dt "cifar10" -ar "resnet18" -mt "alternative" -id "test3" -nw 4 -tp "${type}"
#  fi
#  if [ $1 -eq 1 ]
#  then
#    python stochastic_loss_landscape.py  --sigma "0.003" --batch_size 128 --pruner "global" -pru "0.95" -dt "cifar10" -ar "resnet50" -mt "alternative" -id "test3" -nw 4 -tp "${type}"
#  fi
#  if [ $1 -eq 2 ]
#  then
#    python stochastic_loss_landscape.py  --sigma "0.003" --batch_size 128 --pruner "global" -pru "0.95" -dt "cifar10" -ar "VGG19" -mt "alternative" -id "test3" -nw 4 -tp "${type}"
#  fi
#  if [ $1 -eq 3 ]
#  then
#    python stochastic_loss_landscape.py  --sigma "0.003" --batch_size 128 --pruner "global" -pru "0.9" -dt "cifar100" -ar "resnet18" -mt "alternative" -id "test3" -nw 4 -tp "${type}"
#  fi
#  if [ $1 -eq 4 ]
#  then
#    python stochastic_loss_landscape.py  --sigma "0.001" --batch_size 128 --pruner "global" -pru "0.85" -dt "cifar100" -ar "resnet50" -mt "alternative" -id "test3" -nw 4 -tp "${type}"
#  fi
#  if [ $1 -eq 5 ]
#  then
#    python stochastic_loss_landscape.py --sigma "0.001" --batch_size  128 --pruner "global" -pru "0.8" -dt "cifar100" -ar "VGG19" -mt "alternative" -id "test3" -nw 4 -tp "${type}"
#  fi

#
#
#
#
# python stochastic_loss_landscape.py --sigma "0.005" --batch_size 128 --pruner "global" -pru "0.9" -dt "cifar10" -ar "resnet18" -mt "alternative" -id "test3" -nw 4 -tp "${type}";    python stochastic_loss_landscape.py  --sigma "0.003" --batch_size 128 --pruner "global" -pru "0.95" -dt "cifar10" -ar "resnet50" -mt "alternative" -id "test3" -nw 4 -tp "${type}";python stochastic_loss_landscape.py  --sigma "0.003" --batch_size 128 --pruner "global" -pru "0.95" -dt "cifar10" -ar "VGG19" -mt "alternative" -id "test3" -nw 4 -tp "${type}";    python stochastic_loss_landscape.py  --sigma "0.003" --batch_size 128 --pruner "global" -pru "0.9" -dt "cifar100" -ar "resnet18" -mt "alternative" -id "test3" -nw 4 -tp "${type}";python stochastic_loss_landscape.py  --sigma "0.001" --batch_size 128 --pruner "global" -pru "0.85" -dt "cifar100" -ar "resnet50" -mt "alternative" -id "test3" -nw 4 -tp "${type}";    python stochastic_loss_landscape.py --sigma "0.001" --batch_size  128 --pruner "global" -pru "0.8" -dt "cifar100" -ar "VGG19" -mt "alternative" -id "test3" -nw 4 -tp "${type}"
#

#############################################################
#    Receptive Field and Stochastic Pruning
#############################################################

#python receptive_field_and_stochastic_pruning.py  --model $1 --dataset $2 --num_workers $3 --RF_level $4 --type $5 --folder $6 --pruning_rate $7 --sigma $8

#############################################################
#    Receptive Field and  Second order information optimisers
#############################################################

# python Second_order_Receptive_field.py --experiment 2 --optimiser $1 # --save_folder "$HOME/checkpoints"
#
# python Second_order_Receptive_field.py --experiment 2 --optimiser $1 --save_folder "$HOME/checkpoints"

################ KFAC #####################################
#if [ $6 -eq 1 ]
#then
#python Second_order_Receptive_field.py --lr "0.01" --momentum "0.5" --grad_clip "1" --save 1 --experiment 1 --epochs 100 --batch_size 32 --use_scheduler 1 --use_scheduler_batch 0 --num_workers 4 --optimiser "kfac" --record 1 -dt $1 --model $2 --RF_level $3 --type $4 --name $5 #--save_folder "$HOME/checkpoints"
#fi
#
################# SAM #####################################
#if [ $6 -eq 2 ]
#then
#python Second_order_Receptive_field.py --lr "0.1" --momentum "0.7" --grad_clip "1" --save 1 --experiment 1 --epochs 100 --batch_size 128 --use_scheduler 1 --use_scheduler_batch 0 --num_workers 4 --optimiser "sam" --record 1 -dt $1 --model $2 --RF_level $3 --type $4 --name $5 #--save_folder "$HOME/checkpoints"
#fi

#python hao_models_pruning_test.py --experiment 1



#############################################################
#   Soup Idea applied to stochastic pruning
#############################################################

python main.py --experiment 1 --batch_size 518 --modeltype "alternative" --pruner "global" --population 5 --epochs 10 --pruning_rate  $1 --architecture $2 --sigma $3 --dataset $4
