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

# send mail to this address
#SBATCH --mail-user=sclaam@leeds.ac.uk
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#module load pytorch


#module load python/miniconda
#conda activate ffcv
module load miniforge/
conda activate work


for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

# use here your expected variables
echo "FFCV = $FFCV"
echo "NAME = $NAME"

echo "MODEL = $MODEL"
echo "DATASET= $DATASET"
echo "NUMW= $NUMW"
echo "RFL=$RFL"
echo "TYPE= $TYPE"
echo "FOLDER= $FOLDER"
echo "PR= $PR"
echo "EXPERIMENT= $EXPERIMENT"
echo "FFCV_TRAIN = $FFCV_TRAIN "
echo "FFCV_VAL = $FFCV_VAL "

#    python prune_models.py --name "${NAME}" --ffcv --model $1 --dataset $2 --num_workers $3 --RF_level $4 --type $5 --folder $6 --pruning_rate $7 --experiment $8

echo "EXTRA_VALUES = $EXTRA_VALUES"

#python train_CIFAR10.py --model $1 --dataset $2 --num_workers $3 --RF_level $4 --type $5

#python Second_order_Receptive_field.py --experiment 2 --optimiser "kfac"
#
#python Second_order_Receptive_field.py --experiment 2 --optimiser "sam"
#python -c "import torch;device = 'cuda' if torch.cuda.is_available() else 'cpu';print(device);print('Cuda version with torch: {}'.format(torch.version.cuda))" python -c "import os; print(os.environ)" printf "Start Test \n"
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

#eval "$(conda shell.bash hook)"
#conda activate work
#l=$(which python)
#
#lib_path_of_current_enviroment="${l%%"bin/python"}"
#echo "Ld library ${lib_path_of_current_enviroment}"
#export LD_LIBRARY_PATH="$lib_path_of_current_enviroment/lib":$LD_LIBRARY_PATH

#unset GOMP_CPU_AFFINITY
#unset KMP_AFFINITY

#python -c "import os; print(os.environ)"

#python train_CIFAR10.py  --batch_size 128  --save_folder "/jmain02/home/J2AD014/mtc03/lla98-mtc03/checkpoints" --model $1 --dataset $2 --num_workers $3 --RF_level $4 --type $5 --epochs $6  --name $7 --width $8 --record $9

#python train_CIFAR10.py --resume --save_folder "/jmain02/home/J2AD014/mtc03/lla98-mtc03/checkpoints" --batch_size 128  --model $1 --dataset $2 --num_workers $3 --RF_level $4 --type $5 --epochs $6  --name $7 --width $8 --record $9 --resume_solution "${10}"

#qsub -l h_rt=48:00:00 -l coproc_v100=1 -N "resume_training_Level_3_small_resnet_small_imagenet" resume_run.sh "resnet_small" "small_imagenet" 2 3 "normal" 200 "recording_200" 1 1 "/nobackup/sclaam/checkpoints/resnet_small_normal_small_imagenet_seed.0_rf_level_3_recording_200_test_acc_42.89.pth"

#############################################################
#     Train model
#############################################################

# With FFCV
#python train_CIFAR10.py --ffcv --record_time --record_flops  --batch_size 128  --save_folder "/jmain02/home/J2AD014/mtc03/lla98-mtc03/checkpoints" --model $1 --dataset $2 --num_workers $3 --RF_level $4 --type $5 --epochs $6  --name $7 --width $8 --record $9 --ffcv_train "${10}" --ffcv_val "${11}"

# Without FFCV
#python train_CIFAR10.py --record_time --record_flops  --batch_size 128  --save_folder "/jmain02/home/J2AD014/mtc03/lla98-mtc03/checkpoints" --model $1 --dataset $2 --num_workers $3 --RF_level $4 --type $5 --epochs $6  --name $7 --width $8 --record $9


#

#python3.9 -c "import os; print(os.environ)"
#python3.9 -c "import torch;device = 'cuda' if torch.cuda.is_available() else 'cpu';print(device);print('Cuda version with torch: {}'.format(torch.version.cuda))"
#export LD_LIBRARY_PATH=""
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/jmain02/home/J2AD014/mtc03/lla98-mtc03/.conda/envs/ffcv/lib"
#export PYTHONPATH="/jmain02/home/J2AD014/mtc03/lla98-mtc03/.conda/envs/ffcv/lib/python3.9/site-packages"
#############################################################
#     One shot with specific pruning rate results
#############################################################

  if [[ $FFCV -gt 0 ]]; then
#    export LD_LIBRARY_PATH="/jmain02/home/J2AD014/mtc03/lla98-mtc03/.conda/envs/ffcv/lib"
#    echo "CPUs allocated: $SLURM_JOB_CPUS_PER_NODE"
#    python -c "import os; print(os.environ)"
#    which pythonmo
#     recording_200_ffcv
echo "use FFCV!"

#    python prune_models.py --name "${NAME}" --ffcv --model $1 --dataset $2 --num_workers $3 --RF_level $4 --type $5 --folder $6 --pruning_rate $7 --experiment $8

    python prune_models.py --ffcv --name "${NAME}" --model "${MODEL}" --dataset "${DATASET}" --num_workers "${NUMW}" --RF_level "${RFL}" --type "${TYPE}" --folder "${FOLDER}" --pruning_rate "${PR}"  --experiment "${EXPERIMENT}" --ffcv_train "${FFCV_TRAIN}" --ffcv_val "${FFCV_VAL}" --data_folder "${DATA_FOLDER}" --save_folder "${SAVE_FOLDER}" --input_resolution "${INPUT_RES}" --resize "${RESIZE}" --epochs 10 --record 1
else

#    echo "CPUs allocated: $SLURM_JOB_CPUS_PER_NODE"
#    python -c "import os; print(os.environ)"
#    which python
    echo "Don't use FFCV!"
#./slurm_pruning_run.sh FFCV=0 NAME=recording_200_no_ffcv MODEL=resnet_small DATASET=small_imagenet NUMW=4 RFL=10 TYPE=normal FOLDER=$HOME/checkpoints PR=0.6 EXPERIMENT=1

    python prune_models.py --name "${NAME}" --model "${MODEL}" --dataset "${DATASET}" --num_workers "${NUMW}" --RF_level "${RFL}" --type "${TYPE}" --folder "${FOLDER}" --pruning_rate "${PR}"  --experiment "${EXPERIMENT}" --data_folder "${DATA_FOLDER}" --save_folder "${SAVE_FOLDER}" --input_resolution "${INPUT_RES}" --resize "${RESIZE}" --epochs 10 --record 1
 slurm_pr fi

###########################################################
#   Soup Idea applied to stochastic pruning
#############################################################

#python main.py --experiment 1 --batch_size 518 --modeltype "alternative" --pruner "global" --population 5 --epochs 10 --pruning_rate  $1 --architecture $2 --sigma $3 --dataset $4
