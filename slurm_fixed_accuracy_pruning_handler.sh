#!/bin/bash
# One-shot pruning of the fixed-accuracy checkpoints (slurm_fixed_accuracy_experiments_handler.sh), cifar10,
# 4 models x RF levels 1-4 x 14 learning rates (> 0.003). Pruning rates 0.7, 0.8, 0.9 for every model, plus 0.95,
# 0.97, 0.99 for diracnet50 (it loses little accuracy below 0.99).
#
# Each job runs slurm_fixed_accuracy_pruning_run.sh, which:
#   - one-shot prunes every seed checkpoint found via glob for that (model, rf_level, lr, pruning_rate), recording
#     dense-vs-pruned accuracy (prune_models.py --experiment 1);
#   - computes the saturation of the same pruned model (saturation_calculation.py).
# diracnet50 checkpoints are folded (fold_dirac) before pruning in both scripts.
#
# Checkpoints are read from ${SCRATCH}/fixed_accuracy_experiments; results go to
# ${SCRATCH}/fixed_accuracy_experiments/pruning_results. Run after the training jobs have finished.

models=("resnet50" "vgg19" "resnet18" "diracnet50")
rf_levels=(1 2 3 4)
learning_rates=(0.005 0.007 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.15 0.2)

resolution=32
checkpoints_folder="${SCRATCH}/fixed_accuracy_experiments"
save_folder="${SCRATCH}/fixed_accuracy_experiments/pruning_results"
mkdir -p "${save_folder}"

for model in "${models[@]}"; do
  if [ "${model}" == "diracnet50" ]; then
    pruning_rates=(0.7 0.8 0.9 0.95 0.97 0.99)
  else
    pruning_rates=(0.7 0.8 0.9)
  fi
  for lvl in "${rf_levels[@]}"; do
    for lr in "${learning_rates[@]}"; do

      name="lr_${lr}_fixed_accuracy_no_ffcv"

      for pr in "${pruning_rates[@]}"; do

        job="fixed_acc_${model}_rf${lvl}_cifar10_lr${lr}_pr${pr}_pruning_saturation"

        sbatch --nodes=1 --time=04:00:00 --partition=gpu --gres=gpu:1 --mail-type=ALL \
          --mail-user=l.avendanomunoz@leeds.ac.uk \
          --error="${job}.err" --output="${job}.out" --job-name="${job}" \
          slurm_fixed_accuracy_pruning_run.sh NAME="${name}" MODEL="${model}" DATASET="cifar10" NUMW=8 \
          RFL="${lvl}" TYPE="normal" FOLDER="${checkpoints_folder}" PR="${pr}" DATA_FOLDER="${SCRATCH}/data2" \
          SAVE_FOLDER="${save_folder}" INPUT_RES="${resolution}"

      done
    done
  done
done
