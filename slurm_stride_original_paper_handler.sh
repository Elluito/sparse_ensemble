#!/bin/bash


for lvl in 2 3 4 9 10 11; do                # iterate idxa from 0 to length
for model in "resnet50_stride";do
for dataset in "tiny_imagenet"; do
sbatch --nodes=1 --time=96:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="${model}_lvl_${lvl}_${dataset}.err" --gres=gpu:1 --output="${model}_lvl_${lvl}_${dataset}.out"  --job-name="${model}_lvl_${lvl}_${dataset}" slurm_stride_original_paper_run.sh "${model}" "${dataset}" 4 ${lvl}  "normal" 200 "recording_200" 1 1
done
done
done

for lvl in 3 4 9 10 11; do                # iterate idxa from 0 to length
for model in "resnet50_stride";do
for dataset in "cifar10"; do
sbatch --nodes=1 --time=96:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="${model}_lvl_${lvl}_${dataset}.err" --gres=gpu:1 --output="${model}_lvl_${lvl}_${dataset}.out"  --job-name="${model}_lvl_${lvl}_${dataset}" slurm_stride_original_paper_run.sh "${model}" "${dataset}" 4 ${lvl}  "normal" 200 "recording_200" 1 1
done
done
done

for lvl in 1 2 3 ; do                # iterate idxa from 0 to length
for model in "vgg19_stride";do
for dataset in "cifar10" "tiny_imagenet"; do
sbatch --array=1-2 --nodes=1 --time=96:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="${model}_lvl_${lvl}_${dataset}.err" --gres=gpu:1 --output="${model}_lvl_${lvl}_${dataset}.out"  --job-name="${model}_lvl_${lvl}_${dataset}" slurm_stride_original_paper_run.sh "${model}" "${dataset}" 4 ${lvl}  "normal" 200 "recording_200" 1 1
done
done
done
