#!/bin/bash

res=224
epochs=100
array=0
ffcv=1
for model in "resnet25_small"; do
for dataset in "small_imagenet"; do
#for model in "deep_small_vgg" "resnet25_small"; do # all two models
for lvl in 1; do                # iterate idxA from 0 to length
#for lvl in 1 2 3 4; do                #
#for lvl in 5; do                # iterate idxA from 0 to length
if [ "${array}" -eq 1 ]; then

sbatch --ntasks-per-node=1 --cpus-per-task=16 --nodes=1 --array=1-3 --time=140:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="deep_${model}_lvl_${lvl}_${dataset}_res_${res}.err" --gres=gpu:1 --output="deep_${model}_lvl_${lvl}_${dataset}_res_${res}.output"  --job-name="deep_${model}_lvl_${lvl}_${dataset}_res_${res}" slurm_deep_small_resnet_run.sh "${model}" "${dataset}" 16 ${lvl}  "normal" "${epochs}" "sgd_${epochs}_res_${res}_ffcv" 1 0 "${res}" "${ffcv}"

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

else

sbatch --ntasks-per-node=1 --cpus-per-task=16 --nodes=1  --time=140:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="deep_${model}_lvl_${lvl}_${dataset}_res_${res}.err" --gres=gpu:1 --output="deep_${model}_lvl_${lvl}_${dataset}_res_${res}.output"  --job-name="deep_${model}_lvl_${lvl}_${dataset}_res_${res}" slurm_deep_small_resnet_run.sh "${model}" "${dataset}" 16 ${lvl}  "normal" "${epochs}" "sgd_${epochs}_res_${res}_ffcv" 1 0 "${res}" "${ffcv}"

fi


done
done
done