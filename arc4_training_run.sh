#!/bin/bash
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
#$ -l h_vmem=8G
# -t 2-5
# -pe smp 3
# Send emails when job starts and ends
#$ -m be


#for ARGUMENT in "$@"
#do
#   KEY=$(echo $ARGUMENT | cut -f1 -d=)
#
#   KEY_LENGTH=${#KEY}
#   VALUE="${ARGUMENT:$KEY_LENGTH+1}"
#
#   export "$KEY"="$VALUE"
#done
#
## use here your expected variables
#echo "FFCV = $FFCV"
#echo "NAME = $NAME"
#
#echo "MODEL = $MODEL"
#echo "DATASET= $DATASET"
#echo "NUMW= $NUMW"
#echo "RFL=$RFL"
#echo "TYPE= $TYPE"
#echo "FOLDER= $FOLDER"
#echo "PR= $PR"
#echo "EXPERIMENT= $EXPERIMENT"
#echo "FFCV_TRAIN = $FFCV_TRAIN "
#echo "FFCV_VAL = $FFCV_VAL "
#
##    python prune_models.py --name "${NAME}" --ffcv --model $1 --dataset $2 --num_workers $3 --RF_level $4 --type $5 --folder $6 --pruning_rate $7 --experiment $8
#
#echo "EXTRA_VALUES = $EXTRA_VALUES"


#python train_CIFAR10.py --model $1 --dataset $2 --num_workers $3 --RF_level $4 --type $5

#python Second_order_Receptive_field.py --experiment 2 --optimiser "kfac"
#
#python Second_order_Receptive_field.py --experiment 2 --optimiser "sam"
#export LD_LIBRARY_PATH=""
#l=$(which python)
#lib_path_of_current_enviroment="${l%%python}"
#echo "Ld library ${lib_path_of_current_enviroment}"
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$lib_path_of_current_enviroment
#export PYTHONPATH="/jmain02/home/J2AD014/mtc03/lla98-mtc03/.conda/envs/ffcv/lib/python3.9/site-packages"

conda activate work
which python
unset GOMP_CPU_AFFINITY
unset KMP_AFFINITY

if [ "${11}" -gt 0 ]
  then
  echo "Use FFCV"
#sbatch --nodes=1 --time=07:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk  --error="${model}_${rf_levels[$idxB]}_${dataset}_${pruning_rates[$idxA]}_confidence_ffcv.err" --output="${model}_${rf_levels[$idxB]}_${dataset}_${pruning_rates[$idxA]}_confidence_ffcv.out" --job-name="${model}_${rf_levels[$idxB]}_${dataset}_${pruning_rates[$idxA]}_confidence_ffcv" slurm_confidence_run.sh FFCV="${ffcv}" NAME="${name}" MODEL="${model}" DATASET="${dataset}"  NUMW=4  RFL="${rf_levels[$idxB]}" TYPE="normal" FOLDER="${directory}" PR="${pruning_rates[$idxA]}" EXPERIMENT=1 FFCV_TRAIN="${ffcv_train}" FFCV_VAL="${ffcv_val}" DATA_FOLDER="${data_folder}" OUTPUT_DIR="${output_dir}" TOPK=10

python3.9 train_CIFAR10.py --ffcv --lr "0.1" --batch_size 128  --save_folder "/nobackup/sclaam/training_models_arc4/" --model $1 --dataset $2 --num_workers $3 --RF_level $4 --type $5 --epochs $6  --name $7 --width $8 --record $9 --input_resolution "${10}"  --ffcv_val "${12}" --ffcv_train "${13}" --record_saturation 0 --seed_name "${JOB_ID}.${SGE_TASK_ID}"

#python3.9 ffcv_loaders.py

# slurm_pruning_run.sh FFCV="${ffcv}" NAME="${name}" MODEL="${model}" DATASET="${dataset}"  NUMW=4  RFL="${rf_levels[$idxB]}" TYPE="normal" FOLDER="${directory}" PR="${pruning_rates[$idxA]}" EXPERIMENT=1 FFCV_TRAIN="${ffcv_train}" FFCV_VAL="${ffcv_val}"

#./slurm_pruning_run.sh FFCV=0 NAME=recording_200_no_ffcv MODEL=resnet_small DATASET=small_imagenet NUMW=4 RFL=10 TYPE=normal FOLDER=$HOME/checkpoints PR=0.6 EXPERIMENT=1
else
  echo "Don't use FFCV"

python3.9 train_CIFAR10.py --lr "0.1" --batch_size 128  --save_folder "/nobackup/sclaam/training_models_arc4/" --model $1 --dataset $2 --num_workers $3 --RF_level $4 --type $5 --epochs $6  --name $7 --width $8 --record $9 --input_resolution "${10}" --record_saturation 0 --seed_name "${JOB_ID}.${SGE_TASK_ID}"
#python3.9 ffcv_loaders.py

  fi

#python3.9 Second_order_Receptive_field.py --level $1
#python zero_cost_nas_RF.py
