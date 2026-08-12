#!/bin/bash

# Saturation calculation handler for the diverse pooling models
# (spectral_pool, softpool, mixedpool, lippool) on resnet50/vgg19, cifar10, rf_level 1-4.
#
# Computes saturation on the dense trained checkpoints (pruning_rate=0) found by
# globbing ${directory} for every solution matching model/dataset/rf_level, mirroring
# the pattern used by slurm_saturation_calc_run.sh / saturation_calculation.py.

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

  sbatch --nodes=1 --time=03:00:00 --partition=gpu --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="${model}_${rf_level}_${dataset}_${pruning_rate}_${name}_saturation_ffcv.err" --output="${model}_${rf_level}_${dataset}_${pruning_rate}_${name}_saturation_ffcv.out" --job-name="${model}_${rf_level}_${dataset}_${pruning_rate}_${name}_saturation_ffcv" slurm_diverse_pooling_saturation_calc_run.sh FFCV="${ffcv}" NAME="${name}" MODEL="${model}" DATASET="${dataset}" NUMW=8 RFL="${rf_level}" TYPE="normal" FOLDER="${directory}" PR="${pruning_rate}" FFCV_TRAIN="${ffcv_train}" FFCV_VAL="${ffcv_val}" DATA_FOLDER="${data_folder}" SAVE_FOLDER="${save_folder}"

else

 echo "Don't use FFCV"

 sbatch --nodes=1 --time=03:00:00 --partition=gpu --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="${model}_${rf_level}_${dataset}_${pruning_rate}_${name}_saturation_no_ffcv.err" --output="${model}_${rf_level}_${dataset}_${pruning_rate}_${name}_saturation_no_ffcv.out" --job-name="${model}_${rf_level}_${dataset}_${pruning_rate}_${name}_saturation_no_ffcv" slurm_diverse_pooling_saturation_calc_run.sh FFCV="${ffcv}" NAME="${name}" MODEL="${model}" DATASET="${dataset}" NUMW=8 RFL="${rf_level}" TYPE="normal" FOLDER="${directory}" PR="${pruning_rate}" DATA_FOLDER="${data_folder}" SAVE_FOLDER="${save_folder}"

  fi

}

resolution=32
resize=0
name="recording_diverse_pooling_100_no_ffcv"
checkpoints_folder="${SCRATCH}/diverse_pooling_100"

for lvl in 1 2 3 4; do    # iterate rf_level from 1 to 4
for model in "resnet50_spectral_pool" "resnet50_softpool" "resnet50_mixedpool" "resnet50_lippool" "vgg19_spectral_pool" "vgg19_softpool" "vgg19_mixedpool" "vgg19_lippool"; do
for dataset in "cifar10"; do
for pruning_rate in "0"; do

if [[ "${model}" == vgg19_* ]]; then
  save_folder="${HOME}/sparse_ensemble/saturation_diverse_pooling/cifar10/vgg19"
else
  save_folder="${HOME}/sparse_ensemble/saturation_diverse_pooling/cifar10/resnet50"
fi

run_saturation_calc "${model}" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "${name}" 0 "" "" "${pruning_rate}" "${lvl}"

done
done
done
done
