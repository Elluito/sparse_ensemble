#!/bin/bash
for lvl in 1 2 4 8; do                # iterate idxa from 0 to length
for model in "denset40" "mobilenetv2";do
for dataset in "cifar10" "tiny_imagenet"; do
sbatch --nodes=1 --time=140:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="${model}_lvl_${lvl}_${dataset}.err" --gres=gpu:1 --output="${model}_lvl_${lvl}_${dataset}.out"  --job-name="${model}_lvl_${lvl}_${dataset}" slurm_new_models_original_paper_run.sh "${model}" "${dataset}" 4 ${lvl}  "normal" 200 "recording_200" 1 1
done
done
done