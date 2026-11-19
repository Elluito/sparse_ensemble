#!/bin/bash
# Pruning of the DiracNet-50 / ResNet-18 learning-rate-sweep checkpoints (cifar10, rf_levels 2 and 3, every lr),
# for pruning_rates [0.5, 0.6, 0.7, 0.8, 0.9]. Same design as slurm_learning_rate_pruning_handler.sh.
#
# Each job runs slurm_dirac_resnet18_lr_sweep_pruning_run.sh, which:
#   - one-shot prunes every seed checkpoint found via glob for that (model, rf_level, lr, pruning_rate),
#     recording dense-vs-pruned accuracy (prune_models.py --experiment 1);
#   - computes the saturation of the same pruned model (saturation_calculation.py), which is the saturation used
#     in the ResNet-50 / VGG-19 lr-sweep mediation.
# diracnet50 checkpoints are folded (fold_dirac) before pruning in both scripts.
#
# Run after slurm_dirac_resnet18_lr_sweep_experiments_handler.sh has finished.

learning_rates=(0.0001 0.001 0.003 0.005 0.007 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.15 0.2)
pruning_rates=(0.5 0.6 0.7 0.8 0.9)
rf_levels=(2 3)

resolution=32
checkpoints_folder="${SCRATCH}/learning_rate_experiments"
save_folder="${HOME}/sparse_ensemble/dirac_resnet18_lr_sweep_pruning_results"
mkdir -p "${save_folder}"

for model in "diracnet50" "resnet18"; do
  for lvl in "${rf_levels[@]}"; do
    for lr in "${learning_rates[@]}"; do

      name="lr_${lr}_recording_lr_sweep_no_ffcv"

      for pr in "${pruning_rates[@]}"; do

        job="${model}_rf${lvl}_cifar10_lr${lr}_pr${pr}_pruning_saturation"

        sbatch --nodes=1 --time=04:00:00 --partition=gpu --gres=gpu:1 --mail-type=ALL \
          --mail-user=l.avendanomunoz@leeds.ac.uk \
          --error="${job}.err" --output="${job}.out" --job-name="${job}" \
          slurm_dirac_resnet18_lr_sweep_pruning_run.sh NAME="${name}" MODEL="${model}" DATASET="cifar10" NUMW=8 \
          RFL="${lvl}" TYPE="normal" FOLDER="${checkpoints_folder}" PR="${pr}" DATA_FOLDER="${SCRATCH}/data2" \
          SAVE_FOLDER="${save_folder}" INPUT_RES="${resolution}"

      done
    done
  done
done
