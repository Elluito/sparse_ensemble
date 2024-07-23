#!/bin/bash

run_confidence() {
model=$1
dataset=$2
directory=$3
data_folder=$4
name=$5
ffcv=$6
ffcv_train=$7
ffcv_val=$8
output_dir=$9
#echo "model ${model} and dataset ${dataset}"

pruning_rates=("0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9")
# For resnet18

  if [ "${10}" -gt 0 ]
  then
#      rf_levels=("5" "7" "8" "9")
#      rf_levels=("3" "4" "6" "10")
       rf_levels=("4")

  else
#        rf_levels=("3" "4" "10")
         rf_levels=("11")
  fi

levels_max=${#rf_levels[@]}                                  # Take the length of that array
number_pruning_rates=${#pruning_rates[@]}                            # Take the length of that array
for ((idxA=0; idxA<number_pruning_rates; idxA++)); do                # iterate idxA from 0 to length
for ((idxB=0; idxB<levels_max; idxB++));do              # iterate idxB from 0 to length


if [ "${ffcv}" -gt 0 ]
  then
  echo "Use FFCV"
sbatch --nodes=1 --time=07:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk  --error="${model}_${rf_levels[$idxB]}_${dataset}_${pruning_rates[$idxA]}_confidence_ffcv.err" --output="${model}_${rf_levels[$idxB]}_${dataset}_${pruning_rates[$idxA]}_confidence_ffcv.out" --job-name="${model}_${rf_levels[$idxB]}_${dataset}_${pruning_rates[$idxA]}_confidence_ffcv" slurm_confidence_run.sh FFCV="${ffcv}" NAME="${name}" MODEL="${model}" DATASET="${dataset}"  NUMW=4  RFL="${rf_levels[$idxB]}" TYPE="normal" FOLDER="${directory}" PR="${pruning_rates[$idxA]}" EXPERIMENT=1 FFCV_TRAIN="${ffcv_train}" FFCV_VAL="${ffcv_val}" DATA_FOLDER="${data_folder}" OUTPUT_DIR="${output_dir}" TOPK=10
# slurm_pruning_run.sh FFCV="${ffcv}" NAME="${name}" MODEL="${model}" DATASET="${dataset}"  NUMW=4  RFL="${rf_levels[$idxB]}" TYPE="normal" FOLDER="${directory}" PR="${pruning_rates[$idxA]}" EXPERIMENT=1 FFCV_TRAIN="${ffcv_train}" FFCV_VAL="${ffcv_val}"

#./slurm_pruning_run.sh FFCV=0 NAME=recording_200_no_ffcv MODEL=resnet_small DATASET=small_imagenet NUMW=4 RFL=10 TYPE=normal FOLDER=$HOME/checkpoints PR=0.6 EXPERIMENT=1
else
  echo "Don't use FFCV"

sbatch --nodes=1 --time=07:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk  --error="${model}_${rf_levels[$idxB]}_${dataset}_${pruning_rates[$idxA]}_confidence_no_ffcv.err" --output="${model}_${rf_levels[$idxB]}_${dataset}_${pruning_rates[$idxA]}_confidence_no_ffcv.out" --job-name="${model}_${rf_levels[$idxB]}_${dataset}_${pruning_rates[$idxA]}_confidence_no_ffcv" slurm_confidence_run.sh FFCV="${ffcv}" NAME="${name}" MODEL="${model}" DATASET="${dataset}"  NUMW=0  RFL="${rf_levels[$idxB]}" TYPE="normal" FOLDER="${directory}" PR="${pruning_rates[$idxA]}" EXPERIMENT=1 DATA_FOLDER="${data_folder}" OUTPUT_DIR="${output_dir}" TOPK=10
#   slurm_run.sh  "${model}" "${dataset}" 4  "${rf_levels[$idxB]}" "normal" "${directory}" "${pruning_rates[$idxA]}" 1
#slurm_pruning_run.sh FFCV="${ffcv}" NAME="${name}" MODEL="${model}" DATASET="${dataset}"  NUMW=4  RFL="${rf_levels[$idxB]}" TYPE="normal" FOLDER="${directory}" PR="${pruning_rates[$idxA]}" EXPERIMENT=1
  fi
#echo "${model}_${rf_levels[$idxB]}_${dataset}_${pruning_rates[$idxA]}_pruning in directory ${directory}"
done
done
}
run_confidence "resnet_small" "small_imagenet" "${HOME}/checkpoints_temp" "${HOME}/datasets" "recording_200_ffcv" 0 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" "${HOME}/sparse_ensemble/confidence_and_RF" 1
#run_confidence "resnet_small" "small_imagenet" "${HOME}/checkpoints" "${HOME}/datasets" "recording_200_no_ffcv" 1 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" "${HOME}/sparse_ensemble/confidence_and_RF" 0
#run_confidence "resnet_small" "small_imagenet" "${HOME}/checkpoints" "${HOME}/datasets" "recording_200_ffcv" 0 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" "${HOME}/sparse_ensemble/confidence_and_RF" 1

