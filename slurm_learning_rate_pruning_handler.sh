#!/bin/bash
# Pruning of the learning-rate-sweep checkpoints (vgg19, resnet50 on cifar10,
# rf_levels 1 and 4, all learning rates from the lr sweep), for pruning_rates
# [0.5, 0.6, 0.7, 0.8, 0.9].
#
# Each sbatch job runs slurm_learning_rate_pruning_run.sh, which:
#   - one-shot prunes (no fine-tuning) every seed checkpoint it finds via glob
#     for that (model, rf_level, lr, pruning_rate) combination, recording
#     dense-vs-pruned accuracy (prune_models.py --experiment 1);
#   - then computes saturation of the same pruned model (saturation_calculation.py,
#     which prunes on the fly for its own --pruning_rate branch).
#
# Both results land in the same folder:
# "${HOME}/sparse_ensemble/learning_rate_sweep_pruning_results".
#
# Input checkpoints are read from "${SCRATCH}/learning_rate_experiments", where
# slurm_learning_rate_experiments_handler.sh/_run.sh saved them, and the lr value
# is matched via the same "lr_${lr}_recording_lr_sweep_no_ffcv" name baked into
# every checkpoint filename.

learning_rates=(0.0001 0.001 0.003 0.005 0.007 0.1 0.2)
pruning_rates=(0.5 0.6 0.7 0.8 0.9)
rf_levels=(1 4)

resolution=32
checkpoints_folder="${SCRATCH}/learning_rate_experiments"
save_folder="${HOME}/sparse_ensemble/learning_rate_sweep_pruning_results"

for model in "vgg19" "resnet50"; do
  for lvl in "${rf_levels[@]}"; do
    for lr in "${learning_rates[@]}"; do

      name="lr_${lr}_recording_lr_sweep_no_ffcv"

      for pr in "${pruning_rates[@]}"; do

        sbatch --nodes=1 --time=04:00:00 --partition=gpu --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk \
          --error="${model}_rf${lvl}_cifar10_lr${lr}_pr${pr}_pruning_saturation.err" \
          --output="${model}_rf${lvl}_cifar10_lr${lr}_pr${pr}_pruning_saturation.out" \
          --job-name="${model}_rf${lvl}_cifar10_lr${lr}_pr${pr}_pruning_saturation" \
          slurm_learning_rate_pruning_run.sh NAME="${name}" MODEL="${model}" DATASET="cifar10" NUMW=8 \
          RFL="${lvl}" TYPE="normal" FOLDER="${checkpoints_folder}" PR="${pr}" DATA_FOLDER="${SCRATCH}/data2" \
          SAVE_FOLDER="${save_folder}" INPUT_RES="${resolution}"

      done
    done
  done
done
