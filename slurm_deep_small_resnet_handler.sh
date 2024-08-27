#!/bin/bash
res=110
for lvl in 1; do                # iterate idxA from 0 to length

#sbatch --nodes=1 -array 1-2 --time=140:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="deep_small_resnet_lvl_${lvl}_small_imagenet_res_${res}.err" --gres=gpu:1 --output="deep_small_resnet_lvl_${lvl}_small_imagenet_res_${res}.output"  --job-name="deep_small_resnet_lvl_${lvl}_small_imagenet_res_${res}" slurm_deep_small_resnet_run.sh "resnet25_small" "small_imagenet" 8 ${lvl}  "normal" 200 "recording_200" 1 1 "150"
sbatch --nodes=1 --time=140:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="resnet25_small_lvl_${lvl}_small_imagenet_res_${res}.err" --gres=gpu:1 --output="resnet25_small_lvl_${lvl}_small_imagenet_res_${res}.output"  --job-name="resnet25_small_lvl_${lvl}_small_imagenet_res_${res}" slurm_deep_small_resnet_run.sh "resnet25_small" "small_imagenet" 8 ${lvl}  "normal" 200 "recording_200" 1 1 "150"

done