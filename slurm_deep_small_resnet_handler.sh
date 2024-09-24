#!/bin/bash
res=224
for lvl in 5 6 7 8 9 10; do                # iterate idxA from 0 to length
#for lvl in 4; do                # iterate idxA from 0 to length

#sbatch --nodes=1 -array 1-2 --time=140:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="deep_small_resnet_lvl_${lvl}_small_imagenet_res_${res}.err" --gres=gpu:1 --output="deep_small_resnet_lvl_${lvl}_small_imagenet_res_${res}.output"  --job-name="deep_small_resnet_lvl_${lvl}_small_imagenet_res_${res}" slurm_deep_small_resnet_run.sh "resnet25_small" "small_imagenet" 8 ${lvl}  "normal" 200 "recording_200" 1 1 "150"
sbatch --nodes=1 --time=140:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="resnet_small_lvl_${lvl}_small_imagenet_res_${res}_saturation.err" --gres=gpu:1 --output="resnet_small_lvl_${lvl}_small_imagenet_res_${res}_saturation.output"  --job-name="resnet_small_lvl_${lvl}_small_imagenet_res_${res}_saturation" slurm_deep_small_resnet_run.sh "resnet_small" "small_imagenet" 4 ${lvl}  "normal" 200 "recording_200_ffcv" 1 1 "${res}" 1 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv"

done