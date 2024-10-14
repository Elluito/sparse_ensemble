#!/bin/bash
# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=05:00:00

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
#SBATCH --cpus-per-task=8
which python



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
export LD_LIBRARY_PATH=""
l=$(which python)
lib_path_of_current_enviroment="${l%%python}"
echo "Ld library ${lib_path_of_current_enviroment}"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$lib_path_of_current_enviroment
export PYTHONPATH="/jmain02/home/J2AD014/mtc03/lla98-mtc03/.conda/envs/ffcv/lib/python3.9/site-packages"


  if [ $6 == "ekfac" ]
  then
    # KFac
#python3.9 Second_order_Receptive_field.py --lr "0.01" --momentum "0.5" --grad_clip $6 --save 1 --experiment 1 --record_time 1 --record_flops 1 --record_saturation 1 --epochs 200 --batch_size 128 --use_scheduler 1 --use_scheduler_batch 0 --num_workers 4 --optimiser "ekfac" --record 1 -dt $1 --model $2 --RF_level $3 --type $4 --name $5 --save_folder "$HOME/second_order_experiments"

python3.9 Second_order_Receptive_field.py --lr "0.01" --momentum "0.5"  --save 1 --experiment 1 --batch_size 128 --use_scheduler 1 --use_scheduler_batch 0 --num_workers 2 --optimiser "ekfac" --record 1 --save_folder "$HOME/deep_small_models_saturation" --dataset $1 --model $2 --RF_level $3 --type $4 --name $5 --grad_clip $7 --record_saturation $8  --epochs $9 --input_resolution "${10}"

  fi
  if [ $6 == "sam" ]

    # ASAM
  then
#python3.9 Second_order_Receptive_field.py --lr "0.1" --momentum "0.9" --grad_clip $6 --save 1 --experiment 1 --record_time 1 --record_flops 1 --record_saturation 1 --epochs 100 --batch_size 128 --use_scheduler 1 --use_scheduler_batch 0 --num_workers 0 --optimiser "sam" --record 1 -dt $1 --model $2 --RF_level $3 --type $4 --name $5 --save_folder "$HOME/second_order_experiments"
python3.9 Second_order_Receptive_field.py --lr "0.1" --momentum "0.9" --batch_size 128 --use_scheduler 1 --use_scheduler_batch 0 --num_workers 2  --save 1 --experiment 1 --optimiser "sam" --record 1 --save_folder "$HOME/deep_small_models_saturation"  --dataset $1 --model $2 --RF_level $3 --type $4 --name $5 --grad_clip $7  --record_saturation $8 --epochs $9 --input_resolution "${10}"
  fi

#python3.9 Second_order_Receptive_field.py --level $1
#python zero_cost_nas_RF.py