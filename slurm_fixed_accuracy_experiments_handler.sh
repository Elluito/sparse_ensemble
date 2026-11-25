#!/bin/bash
# Fixed-accuracy training (cifar10): every model is trained with each learning rate LARGER than 0.003 until its test
# accuracy first reaches the dense accuracy of the lr = 0.003 sweep runs for that model and RF level
# (dense_acc_lr_0003.csv), then training stops (train_CIFAR10.py --experiment 5, train_until_accuracy).
#
# Purpose: compare learning rates at the SAME dense accuracy, so differences in saturation and prunability are not
# just differences in how well the dense model is trained.
#
# 4 models x 4 RF levels x 14 learning rates, 3 seeds each (array tasks) = 672 runs.
# --epochs 100 is the maximum and sets the cosine schedule (T_max = 100) as in the lr sweep, so a run that stops at
# epoch k followed the same lr trajectory as a full sweep run up to epoch k. Saturation is recorded every epoch, so
# the last row of the saturation csv is the dense saturation of the saved checkpoint.
#
# The lr is baked into the run name ("lr_${lr}_fixed_accuracy_no_ffcv"). Outputs (checkpoint at the stopping epoch,
# accuracy / saturation csvs, *_fixed_accuracy_summary.csv with the stopping epoch and whether the target was
# reached) go to ${SCRATCH}/fixed_accuracy_experiments.

models=("resnet50" "vgg19" "resnet18" "diracnet50")
rf_levels=(1 2 3 4)
learning_rates=(0.005 0.007 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.15 0.2)

dataset="cifar10"
resolution=32
num_workers=8
max_epochs=100
walltime="47:00:00"
save_folder="${SCRATCH}/fixed_accuracy_experiments"
target_csv="dense_acc_lr_0003.csv"
mkdir -p "${save_folder}"

for model in "${models[@]}"; do
  for lvl in "${rf_levels[@]}"; do
    for lr in "${learning_rates[@]}"; do

      name="lr_${lr}_fixed_accuracy_no_ffcv"
      job="fixed_acc_${model}_${dataset}_rf${lvl}_lr${lr}"

      sbatch --nodes=1 --time="${walltime}" --array=1-3 --partition=gpu --gres=gpu:1 --mail-type=all \
        --mail-user=l.avendanomunoz@leeds.ac.uk \
        --error="${job}_seed_%a.err" --output="${job}_seed_%a.out" --job-name="${job}" \
        slurm_fixed_accuracy_experiments_run.sh "${model}" "${dataset}" "${num_workers}" "${lvl}" \
        "${max_epochs}" "${name}" "${lr}" "${resolution}" "${save_folder}" "${target_csv}"

    done
  done
done
