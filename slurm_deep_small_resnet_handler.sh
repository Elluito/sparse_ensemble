#!/bin/bash

#res=224
#epochs=100
#array=1
#ffcv=0
#for model in "resnet25_small"; do
#for dataset in "small_imagenet"; do
##for model in "deep_small_vgg" "resnet25_small"; do # all two models
#for lvl in 11 12 13; do                # iterate idxA from 0 to length
##for lvl in 1 2 3 4; do                #
##for lvl in 5; do                # iterate idxA from 0 to length
#if [ "${array}" -eq 1 ]; then
#
#sbatch --ntasks-per-node=1 --cpus-per-task=16 --nodes=1 --array=1-3 --time=140:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="deep_${model}_lvl_${lvl}_${dataset}_res_${res}.err" --gres=gpu:1 --output="deep_${model}_lvl_${lvl}_${dataset}_res_${res}.output"  --job-name="deep_${model}_lvl_${lvl}_${dataset}_res_${res}" slurm_deep_small_resnet_run.sh "${model}" "${dataset}" 16 ${lvl}  "normal" "${epochs}" "sgd_${epochs}_res_${res}" 1 0 "${res}" "${ffcv}"
#
##SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=8
#
#else
#
#sbatch --ntasks-per-node=1 --cpus-per-task=16 --nodes=1  --time=140:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="deep_${model}_lvl_${lvl}_${dataset}_res_${res}.err" --gres=gpu:1 --output="deep_${model}_lvl_${lvl}_${dataset}_res_${res}.output"  --job-name="deep_${model}_lvl_${lvl}_${dataset}_res_${res}" slurm_deep_small_resnet_run.sh "${model}" "${dataset}" 16 ${lvl}  "normal" "${epochs}" "sgd_${epochs}_res_${res}" 1 0 "${res}" "${ffcv}"
#
#fi
#
#
#done
#done
#done

res=224
epochs=100
array=1
ffcv=0
ffcv_train="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv"
ffcv_test="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv"
#save_folder="/jmain02/home/J2AD014/mtc03/lla98-mtc03/deep_small_models_ffcv"
save_folder="/jmain02/home/J2AD014/mtc03/lla98-mtc03/deep_small_models_2"

if [ "${ffcv}" -eq 1 ]; then
  ffcv_string="ffcv"
  else
  ffcv_string="no_ffcv"
  fi
for model in "resnet25_small"; do
for dataset in "small_imagenet"; do
#for model in "deep_small_vgg" "resnet25_small"; do # all two models
for lvl in 11 12 13; do                # iterate idxA from 0 to length
#for lvl in 2 3; do                #
#for lvl in 5; do                iterate idxA from 0 to length
jname="deep_${model}_lvl_${lvl}_${dataset}_res_${res}_${ffcv_string}"
if [ "${array}" -eq 1 ]; then

sbatch --ntasks-per-node=1 --cpus-per-task=16 --nodes=1 --array=1-4 --time=140:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="${jname}.err" --gres=gpu:1 --output="${jname}.out"  --job-name="${jname}" slurm_deep_small_resnet_run.sh "${model}" "${dataset}" 16 ${lvl}  "normal" "${epochs}" "sgd_${epochs}_res_${res}_${ffcv_string}" 1 0 "${res}" "${ffcv}" "${ffcv_train}" "${ffcv_test}" "${save_folder}"

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

else

sbatch --ntasks-per-node=1 --cpus-per-task=16 --nodes=1  --time=140:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="${jname}.err" --gres=gpu:1 --output="${jname}.out"  --job-name="${jname}" slurm_deep_small_resnet_run.sh "${model}" "${dataset}" 16 ${lvl}  "normal" "${epochs}" "sgd_${epochs}_res_${res}_${ffcv_string}" 1 0 "${res}" "${ffcv}" "${ffcv_train}" "${ffcv_test}" "${save_folder}"


fi


done
done
done
