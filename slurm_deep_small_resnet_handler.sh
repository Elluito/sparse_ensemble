#!/bin/bash

for lvl in 2 3 4 5 6 7; do                # iterate idxA from 0 to length

sbatch --nodes=1 --time=96:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="deep_small_resnet_lvl_${lvl}_small_imagenet.err" --gres=gpu:1 --output="deep_small_resnet_lvl_${lvl}_small_imagenet.output"  --job-name="deep_small_resnet_lv_${lvl}_small_imagenet" slurm_deep_small_resnet_run.sh "deep_resnet_small" "small_imagenet" 4 ${lvl}  "normal" 200 "recording_200" 1 1

done