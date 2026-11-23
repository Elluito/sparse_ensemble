#!/bin/bash
# High-pruning-rate pruning of the DiracNet-50 learning-rate-sweep checkpoints (cifar10, rf_levels 1-4, every lr),
# for pruning_rates [0.95, 0.97, 0.99]. At 0.5-0.9 DiracNet-50 barely loses accuracy (mean scaled delta 0.06 at 0.9),
# so these rates are meant to give Δ enough spread for the saturation mediation.
#
# Each job runs slurm_dirac_resnet18_lr_sweep_pruning_run.sh (unchanged), which:
#   - folds every diracnet50 seed checkpoint found via glob for that (rf_level, lr) and one-shot prunes it globally
#     at PR, recording dense-vs-pruned accuracy (prune_models.py --experiment 1, --dirac_width 4);
#   - computes the saturation of the same pruned, BN-recalibrated model (saturation_calculation.py).
# Results go to the same folder as the 0.5-0.9 runs; every output filename contains the pruning rate, so nothing is
# overwritten. Build the per-rate CSVs afterwards with build_dirac_resnet18_lr_sweep_pruning_saturation_combined.py.
#
# 4 rf_levels x 17 lrs x 3 pruning rates = 204 jobs (each job prunes the 3 seeds of its combination).

learning_rates=(0.0001 0.001 0.003 0.005 0.007 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.15 0.2)
pruning_rates=(0.95 0.97 0.99)
rf_levels=(1 2 3 4)
model="diracnet50"

resolution=32
checkpoints_folder="${SCRATCH}/learning_rate_experiments"
save_folder="${HOME}/sparse_ensemble/dirac_resnet18_lr_sweep_pruning_results"
mkdir -p "${save_folder}"

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
