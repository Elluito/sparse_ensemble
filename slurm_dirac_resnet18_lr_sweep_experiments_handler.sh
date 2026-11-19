#!/bin/bash
# Learning-rate sweep for the DiracNet-50 / ResNet-18 controls (cifar10), same design as the ResNet-50 / VGG-19
# sweep (slurm_learning_rate_experiments_handler.sh): rf_levels 2 and 3, every lr used in that sweep, 3 seeds
# (array tasks) per (model, rf_level, lr), 100 epochs, saturation recorded every epoch.
#
# Purpose: lr moves saturation at a FIXED kernel size, which the 10-seed lr=0.1 runs cannot do (saturation and
# kernel size were almost collinear there), so the mediation kernel size -> saturation -> Δ can be estimated.
#
# The lr is baked into the run name ("lr_${lr}_recording_lr_sweep_no_ffcv", same as the original sweep), so it is in
# every output file name. Checkpoints / accuracy / saturation go to ${SCRATCH}/learning_rate_experiments, next to
# the ResNet-50 / VGG-19 runs (file names start with the model, so the globs do not mix them).
# Each array task gets its own seed name (time stamp) from train_CIFAR10.py; logs are per seed (%a).

rf_levels=(2 3)
learning_rates=(0.0001 0.001 0.003 0.005 0.007 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.15 0.2)

dataset="cifar10"
resolution=32
num_workers=8
epochs=100
walltime="47:00:00"
save_folder="${SCRATCH}/learning_rate_experiments"

for model in "diracnet50" "resnet18"; do
  for lvl in "${rf_levels[@]}"; do
    for lr in "${learning_rates[@]}"; do

      name="lr_${lr}_recording_lr_sweep_no_ffcv"
      job="lr_sweep_${model}_${dataset}_rf${lvl}_lr${lr}"

      sbatch --nodes=1 --time="${walltime}" --array=1-3 --partition=gpu --gres=gpu:1 --mail-type=all \
        --mail-user=l.avendanomunoz@leeds.ac.uk \
        --error="${job}_seed_%a.err" --output="${job}_seed_%a.out" --job-name="${job}" \
        slurm_dirac_resnet18_lr_sweep_experiments_run.sh "${model}" "${dataset}" "${num_workers}" "${lvl}" \
        "${epochs}" "${name}" "${lr}" "${resolution}" "${save_folder}"

    done
  done
done
