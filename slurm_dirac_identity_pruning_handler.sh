#!/bin/bash
# How much of DiracNet-50's identity (Dirac centre-tap) weight survives pruning, vs the learned weights.
# One CPU job per RF level; each job reads every lr / seed checkpoint of that level from
# ${SCRATCH}/learning_rate_experiments and writes
#   dirac_identity_pruning_results/diracnet50_identity_pruning_{summary,per_layer}_rf_<lvl>.csv
# Pruning rates cover the 0.5-0.9 runs already done and the 0.95-0.99 runs of
# slurm_diracnet50_lr_sweep_high_pr_pruning_handler.sh. The masks are recomputed exactly as prune_models.py does
# (fold_dirac + global L1 pruning, conv1 / linear excluded), so this does not need those pruning jobs to have run.
#
# No --partition / --gres: no GPU is needed. Add your cluster's CPU partition if the default is not suitable.

rf_levels=(1 2 3 4)
pruning_rates="0.5 0.6 0.7 0.8 0.9 0.95 0.97 0.99"
checkpoints_folder="${SCRATCH}/learning_rate_experiments"
save_folder="${HOME}/sparse_ensemble/dirac_identity_pruning_results"
mkdir -p "${save_folder}"

for lvl in "${rf_levels[@]}"; do
  job="diracnet50_rf${lvl}_identity_pruning"
  sbatch --error="${job}.err" --output="${job}.out" --job-name="${job}" \
    slurm_dirac_identity_pruning_run.sh RFL="${lvl}" FOLDER="${checkpoints_folder}" SAVE_FOLDER="${save_folder}" \
    PRS="${pruning_rates}"
done
