#!/bin/bash
# Learning-rate sweep: SGD training of vgg19 and resnet50 on cifar10 and
# tiny_imagenet, across rf_levels 1 and 4, for lr in
# [0.0001, 0.001, 0.003, 0.005, 0.007, 0.1, 0.2].
#
# Each (model, dataset, rf_level, lr) combination is submitted as one
# --array=1-3 job (3 seeds per combination/array-task). Results (accuracy,
# flops, saturation) land in "${SCRATCH}/learning_rate_experiments", and the
# lr value is baked into the run "name" so it shows up in every output
# filename produced by train_CIFAR10.py.

rf_levels=(1 4)
learning_rates=(0.0001 0.001 0.003 0.005 0.007 0.1 0.2)

for model in "vgg19" "resnet50"; do
  for dataset in "cifar10" "tiny_imagenet"; do

    if [ "${dataset}" == "cifar10" ]; then
      resolution=32
      num_workers=8
      epochs=100
      walltime="47:00:00"
    else
      resolution=64
      num_workers=8
      epochs=100
      walltime="47:00:00"
    fi

    for lvl in "${rf_levels[@]}"; do
      for lr in "${learning_rates[@]}"; do

        name="lr_${lr}_recording_lr_sweep_no_ffcv"

        sbatch --nodes=1 --time="${walltime}" --array=1-3 --partition=gpu --gres=gpu:1 --mail-type=all --mail-user=pznz720@leeds.ac.uk \
          --error="lr_sweep_${model}_${dataset}_rf${lvl}_lr${lr}.err" \
          --output="lr_sweep_${model}_${dataset}_rf${lvl}_lr${lr}.out" \
          --job-name="lr_sweep_${model}_${dataset}_rf${lvl}_lr${lr}" \
          slurm_learning_rate_experiments_run.sh "${model}" "${dataset}" "${num_workers}" "${lvl}" "${epochs}" "${name}" "${lr}" "${resolution}"

      done
    done
  done
done
