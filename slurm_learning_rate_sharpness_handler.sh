#!/bin/bash
# Sharpness (m_apgd_linf) handler for the learning-rate-sweep checkpoints
# (vgg19, resnet50 on cifar10, rf_levels 1 and 4, all 17 learning rates from
# the sweep), mirroring slurm_sharpness_calc_handler.sh's run_sharpness.
#
# Checkpoints are read from "${SCRATCH}/learning_rate_experiments" (matched via
# the same "lr_${lr}_recording_lr_sweep_no_ffcv" name baked into every checkpoint
# filename); calculate_sharpness.py globs every seed found for that combination
# in one job (no --array needed). Results land in
# "${HOME}/sparse_ensemble/sharpness_lr_sweep_results".

run_sharpness() {

model=$1
dataset=$2
directory=$3
data_folder=$4
save_folder=$5
name=$6
rf_level=$7
algo=$8

sbatch --nodes=1 --time=03:00:00 --partition=gpu --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk \
  --error="${model}_${rf_level}_${dataset}_${name}_sharpness.err" \
  --output="${model}_${rf_level}_${dataset}_${name}_sharpness.out" \
  --job-name="${model}_${rf_level}_${dataset}_${name}_sharpness" \
  slurm_learning_rate_sharpness_run.sh NAME="${name}" MODEL="${model}" DATASET="${dataset}" NUMW=8 \
  RFL="${rf_level}" TYPE="normal" FOLDER="${directory}" ALGO="${algo}" DATA_FOLDER="${data_folder}" \
  SAVE_FOLDER="${save_folder}"

}

learning_rates=(0.0001 0.001 0.003 0.005 0.007 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.15 0.2)
rf_levels=(1 4)

checkpoints_folder="${SCRATCH}/learning_rate_experiments"
save_folder="${HOME}/sparse_ensemble/sharpness_lr_sweep_results"
algo="m_apgd_linf"

for model in "vgg19" "resnet50"; do
  for lvl in "${rf_levels[@]}"; do
    for lr in "${learning_rates[@]}"; do

      name="lr_${lr}_recording_lr_sweep_no_ffcv"

      run_sharpness "${model}" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "${name}" "${lvl}" "${algo}"

    done
  done
done
