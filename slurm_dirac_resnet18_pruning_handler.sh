#!/bin/bash

# Pruning + fine-tuning handler for the DiracNet-50 / ResNet-18 control models, cifar10, rf_level 1-4
# (trained with slurm_dirac_resnet18_experiments_handler.sh, 10 seeds each).
#
# Two experiments are launched via prune_models.py:
#   - experiment=1: one-shot pruning. Globs every *_test_acc_*.pth solution matching
#     model/dataset/rf_level/name inside the checkpoints folder and prunes each one (no fine-tuning).
#   - experiment=2: prune + fine-tune (10 epochs, set in slurm_dirac_resnet18_pruning_run.sh) one solution per job.
#     The solutions are found with a glob at submission time, so run this after training has finished.
#
# Set RUN_ONE_SHOT / RUN_FINETUNE to 0 to skip either experiment.

RUN_ONE_SHOT=1
RUN_FINETUNE=0

resolution=32
resize=0
data_folder="${SCRATCH}/data2"
dataset="cifar10"

run_pruning() {
model=$1
directory=$2
save_folder=$3
name=$4
pruning_rate=$5
rf_level=$6

job="${model}_${rf_level}_${dataset}_${pruning_rate}_pruning_no_ffcv"
sbatch --nodes=1 --time=48:00:00 --partition=gpu --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="${job}.err" --output="${job}.out" --job-name="${job}" slurm_dirac_resnet18_pruning_run.sh NAME="${name}" MODEL="${model}" DATASET="${dataset}" NUMW=8 RFL="${rf_level}" TYPE="normal" FOLDER="${directory}" PR="${pruning_rate}" EXPERIMENT=1 SOLUTION="" DATA_FOLDER="${data_folder}" SAVE_FOLDER="${save_folder}" INPUT_RES="${resolution}" RESIZE="${resize}"
}

run_pruning_finetune() {
model=$1
directory=$2
save_folder=$3
name=$4
pruning_rate=$5
rf_level=$6
solution=$7
seed_index=$8

job="${model}_${rf_level}_${dataset}_${pruning_rate}_finetune_no_ffcv_seed_${seed_index}"
sbatch --nodes=1 --time=03:00:00 --partition=gpu --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="${job}.err" --output="${job}.out" --job-name="${job}" slurm_dirac_resnet18_pruning_run.sh NAME="${name}" MODEL="${model}" DATASET="${dataset}" NUMW=8 RFL="${rf_level}" TYPE="normal" FOLDER="${directory}" PR="${pruning_rate}" EXPERIMENT=2 SOLUTION="${solution}" DATA_FOLDER="${data_folder}" SAVE_FOLDER="${save_folder}" INPUT_RES="${resolution}" RESIZE="${resize}"
}

# model | checkpoints folder (training output) | pruning results folder | run name used for training
configs=(
"diracnet50|${SCRATCH}/dirac_models_experiments|${HOME}/sparse_ensemble/dirac_pruning_results|recording_dirac_100_no_ffcv"
"resnet18|${SCRATCH}/resnet18_models_experiments|${HOME}/sparse_ensemble/resnet18_pruning_results|recording_resnet18_100_no_ffcv"
)

for config in "${configs[@]}"; do
IFS="|" read -r model checkpoints_folder save_folder name <<< "${config}"
mkdir -p "${save_folder}"

#############################################################
# Experiment 1: one-shot pruning (all seeds found via glob in the checkpoints folder)
#############################################################
if [ "${RUN_ONE_SHOT}" -gt 0 ]; then
for lvl in 1 2 3 4; do    # iterate rf_level from 1 to 4
for pruning_rate in "0.7" "0.8" "0.9"; do

run_pruning "${model}" "${checkpoints_folder}" "${save_folder}" "${name}" "${pruning_rate}" "${lvl}"

done
done
fi

#############################################################
# Experiment 2: prune + fine-tune (10 epochs), one job per seed
# Collapsed training runs (test_acc_10.0, chance level) are skipped, as in the diverse pooling runs.
#############################################################
if [ "${RUN_FINETUNE}" -gt 0 ]; then
pruning_rate=0.9
for lvl in 1 2 3 4; do
seed_index=0
for path in "${checkpoints_folder}/${model}_normal_${dataset}_"*"_rf_level_${lvl}_${name}_test_acc_"*.pth; do
  if [ ! -e "${path}" ]; then
    echo "WARNING: no checkpoints found for ${model} rf_level ${lvl} in ${checkpoints_folder}"
    break
  fi
  solution=$(basename "${path}")
  if [[ "${solution}" == *"_test_acc_10.0.pth" ]]; then
    echo "Skipping collapsed run ${solution}"
    continue
  fi
  seed_index=$((seed_index + 1))

  run_pruning_finetune "${model}" "${checkpoints_folder}" "${save_folder}" "${name}" "${pruning_rate}" "${lvl}" "${solution}" "${seed_index}"

done
done
fi

done
