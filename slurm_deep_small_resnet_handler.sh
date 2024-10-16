#!/bin/bash

res=224
epochs=100
array=0
#for lvl in 5 6 7 8 10; do                # iterate idxA from 0 to length
for model in "resnet25_small"; do
for dataset in "cifar10"; do
#for model in "deep_small_vgg" "resnet25_small"; do # all two models
#for lvl in 5 6 7 8 10; do                #
for lvl in 5; do                # iterate idxA from 0 to length
if [ "${array}" -eq 1 ]; then

sbatch --nodes=1 --array=1-3 --time=140:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="deep_${model}_lvl_${lvl}_${dataset}_res_${res}.err" --gres=gpu:1 --output="deep_${model}_lvl_${lvl}_${dataset}_res_${res}.output"  --job-name="deep_${model}_lvl_${lvl}_${dataset}_res_${res}" slurm_deep_small_resnet_run.sh "${model}" "${dataset}" 2 ${lvl}  "normal" "${epochs}" "sgd_${epochs}_res_${res}" 1 0 "${res}"

else

sbatch --nodes=1  --time=140:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="deep_${model}_lvl_${lvl}_${dataset}_res_${res}.err" --gres=gpu:1 --output="deep_${model}_lvl_${lvl}_${dataset}_res_${res}.output"  --job-name="deep_${model}_lvl_${lvl}_${dataset}_res_${res}" slurm_deep_small_resnet_run.sh "${model}" "${dataset}" 2 ${lvl}  "normal" "${epochs}" "sgd_${epochs}_res_${res}" 1 0 "${res}"
fi

#sbatch --nodes=1 --time=140:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="deep_{}_lvl_${lvl}_small_imagenet_res_${res}.err" --gres=gpu:1 --output="deep_small_resnet_lvl_${lvl}_small_imagenet_res_${res}.output"  --job-name="deep_small_resnet_lvl_${lvl}_small_imagenet_res_${res}" slurm_deep_small_resnet_run.sh "resnet25_small" "small_imagenet" 8 ${lvl}  "normal" "${epochs}" "recording_${epochs}" 1 1 "${res}"
#sbatch --nodes=1 --time=140:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="resnet_small_lvl_${lvl}_small_imagenet_res_${res}_saturation_${epochs}.err" --gres=gpu:1 --output="resnet_small_lvl_${lvl}_small_imagenet_res_${res}_saturation_${epochs}.output"  --job-name="resnet_small_lvl_${lvl}_small_imagenet_res_${res}_saturation_${epochs}" slurm_deep_small_resnet_run.sh "resnet_small" "small_imagenet" 4 ${lvl}  "normal" 100 "recording_100_res_${res}_ffcv" 1 1 "${res}" 1 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv"
done
done
done