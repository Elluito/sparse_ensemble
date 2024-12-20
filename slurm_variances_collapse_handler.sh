#!/bin/bash



run_pruning() {
model=$1
dataset=$2
directory=$3
data_folder=$4
save_folder=$5
name=$6
ffcv=$7
ffcv_train=$8
ffcv_val=$9
#pruning_rates=("0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9")
#pruning_rates=("0.5")
pruning_rate="${10}"
rf_level="${11}"
resolution="${12}"
resize="${13}"
# For resnet18
#
#  if [ "${10}" -gt 0 ]
#  then
##      rf_levels=("5" "7" "8" "9")
#       rf_levels=("5")
#
#  else
#        rf_levels=("3" "4" "6" "10")
##         rf_levels=("6")
#  fi

#levels_max=${#rf_levels[@]}                                  # Take the length of that array
#number_pruning_rates=${#pruning_rates[@]}                            # Take the length of that array
#for ((idxA=0; idxA<number_pruning_rates; idxA++)); do                # iterate idxA from 0 to length
#for ((idxB=0; idxB<levels_max; idxB++));do              # iterate idxB from 0 to length


if [ "${ffcv}" -gt 0 ]
  then
  echo "Use FFCV"
sbatch --nodes=1 --time=03:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk  --error="${model}_${dataset}_${pruning_rate]}.err" --output="${model}_${dataset}_${pruning_rates}.out" --job-name="${model}_${dataset}_${pruning_rate}"   slurm_pruning_run.sh FFCV="${ffcv}" NAME="${name}" MODEL="${model}" DATASET="${dataset}"  NUMW=16  RFL="${rf_level}" TYPE="normal" FOLDER="${directory}" PR="${pruning_rate}" EXPERIMENT=1 FFCV_TRAIN="${ffcv_train}" FFCV_VAL="${ffcv_val}" DATA_FOLDER="${data_folder}" SAVE_FOLDER="${save_folder}" INPUT_RES="${resolution}" RESIZE="${resize}"

else
 echo "Don't use FFCV"
 sbatch --nodes=1 --time=03:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk  --error="${model}_${dataset}_${pruning_rate}_pruning_no_ffcv.err" --output="${model}_${rf_level}_${dataset}_${pruning_rate}.out" --job-name="${model}_${dataset}_${pruning_rate}" slurm_pruning_run.sh FFCV="${ffcv}" NAME="${name}" MODEL="${model}" DATASET="${dataset}"  NUMW=16  RFL="${rf_level}" TYPE="normal" FOLDER="${directory}" PR="${pruning_rate}" EXPERIMENT=1 DATA_FOLDER="${data_folder}" SAVE_FOLDER="${save_folder}" INPUT_RES="${resolution}" RESIZE="${resize}"
  fi
#done
#done

}




    solutions_list =["trained_models/cifar10/resnet18_cifar10_traditional_train_valacc=95,370.pth","trained_models/cifar100/resnet18_cifar100_traditional_train.pth","trained_models/cifar10/VGG19_cifar10_traditional_train_valacc=93,57.pth","trained_models/cifar100/vgg19_cifar100_traditional_train.pth","trained_models/cifar10/resnet50_cifar10.pth","trained_models/cifar100/resnet50_cifar100.pth"]
    models_list= ["resnet18","resnet18","vgg19","vgg19","resnet50","resnet50"]
    datasets_list=["cifar10","cifar100"]*3
    pr_list = [0.9,0.9,0.95,0.8,0.95,0.85]
    sigma_list = [0.005,0.003,0.003,0.001,0.003,0.001]




