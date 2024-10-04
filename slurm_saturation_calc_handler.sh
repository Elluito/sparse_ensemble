#!/bin/bash

run_saturation_calc() {

model=$1
dataset=$2
directory=$3
data_folder=$4
save_folder=$5
name=$6
ffcv=$7
ffcv_train=$8
ffcv_val=$9

pruning_rate="${10}"
rf_level="${11}"

if [ "${ffcv}" -gt 0 ]
  then

  echo "Use FFCV"

  sbatch --nodes=1 --time=03:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk  --error="${model}_${rf_level}_${dataset}_${pruning_rate}_${name}_saturation_ffcv.err" --output="${model}_${rf_level}_${dataset}_${pruning_rate}_${name}_ffcv.out" --job-name="${model}_${rf_level}_${dataset}_${pruning_rate}_${name}_no_pr_ffcv" slurm_inter_layer_run.sh FFCV="${ffcv}" NAME="${name}" MODEL="${model}" DATASET="${dataset}"  NUMW=0  RFL="${rf_level}" TYPE="normal" FOLDER="${directory}" PR="${pruning_rate}" EXPERIMENT=6 FFCV_TRAIN="${ffcv_train}" FFCV_VAL="${ffcv_val}" DATA_FOLDER="${data_folder}" SAVE_FOLDER="${save_folder}"

else

 echo "Don't use FFCV"

 sbatch --nodes=1 --time=03:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk  --error="${model}_${rf_level}_${dataset}_${pruning_rate}_${name}_saturation_ffcv.err" --output="${model}_${rf_level}_${dataset}_${pruning_rate}_${name}_no_ffcv.out" --job-name="${model}_${rf_level}_${dataset}_${pruning_rate}_${name}_no_ffcv" slurm_inter_layer_run.sh FFCV="${ffcv}" NAME="${name}" MODEL="${model}" DATASET="${dataset}"  NUMW=0  RFL="${rf_level}" TYPE="normal" FOLDER="${directory}" PR="${pruning_rate}" EXPERIMENT=6 DATA_FOLDER="${data_folder}" SAVE_FOLDER="${save_folder}"
  fi

#done
#done

}



for model in "resnet_small"; do
for dataset in "small_imagenet"; do
for pruning_rate in "0" "0.9"; do
for rf_level in "3" "4" "5" "6" "7" "8" "9" "10"; do
#for rf_level in "1"; do

run_saturation_calc "${model}" "${dataset}" "${HOME}/resnet_small_saturation" "${HOME}/datasets" "${HOME}/sparse_ensemble/saturation_results/small_imagenet/resnet_small/SGD" "no_recording_200" 0 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" "${pruning_rate}" "${rf_level}"


done
done
done
done
