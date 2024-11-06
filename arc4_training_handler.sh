#!/bin/bash

#sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="ekfac_optim_rf_1_cifar10_vgg19_gc_${grad_clip}.err" --output="ekfac_optim_rf_1_cifar10_vgg19_gc_${grad_clip}.output"  --job-name="ekfac_optim_rf_1_cifar10_vgg19_gc_${grad_clip}" slurm_2nd_order_run.sh "cifar10" "vgg19" "1" "normal" "ekfac_optim_hyper_100_gc_${grad_clip}" "1" grad_clip
#sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="ekfac_optim_rf_2_cifar10_vgg19_gc_${grad_clip}.err" --output="ekfac_optim_rf_2_cifar10_vgg19_gc_${grad_clip}.output" --job-name="ekfac_optim_rf_2_cifar10_vgg19_gc_${grad_clip}" slurm_2nd_order_run.sh "cifar10" "vgg19" "2" "normal" "ekfac_optim_hyper_100_gc_${grad_clip}" "1" grad_clip
#sbatch --nodes=1 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="ekfac_optim_rf_3_cifar10_vgg19_gc_${grad_clip}.err" --output="ekfac_optim_rf_3_cifar10_vgg19_gc_${grad_clip}.output"  --job-name="ekfac_optim_rf_3_cifar10_vgg19_gc_${grad_clip}" slurm_2nd_order_run.sh "cifar10" "vgg19" "3" "normal" "ekfac_optim_hyper_saturation_200_gc_${grad_clip}" "1" grad_clip
#sbatch --nodes=1 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="ekfac_optim_rf_4_cifar10_vgg19_gc_${grad_clip}.err" --output="ekfac_optim_rf_4_cifar10_vgg19_gc_${grad_clip}.output" --job-name="ekfac_optim_rf_4_cifar10_vgg19_gc_${grad_clip}" slurm_2nd_order_run.sh "cifar10" "vgg19" "4" "normal" "ekfac_optim_hyper_saturation_200_gc_${grad_clip}" "1" grad_clip
run_2nd_order_experiment(){
model=$1
dataset=$2
algorithm=$3
name=$4
epochs=$5
grad_clip=$6
RF_level=$7
record_saturation=$8
array=$9
input_res="${10}"

if [ "${array}" -eq 1 ]; then

#sbatch --nodes=1 --array=1-4 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="${algorithm}_rf_${RF_level}_${dataset}_${model}_gc_${grad_clip}.err" --output="${algorithm}_rf_${RF_level}_${dataset}_${model}_gc_${grad_clip}.out"  --job-name="${algorithm}_rf_${RF_level}_${dataset}_${model}_gc_${grad_clip}" slurm_2nd_order_run.sh "${dataset}" "${model}" "${RF_level}" "normal" "${name}" "${algorithm}" "${grad_clip}" "${record_saturation}" "${epochs}" "${input_res}"

qsub -l h_rt=44:00:00 -t 1-3 -l coproc_v100=1  -N  "${algorithm}_rf_${RF_level}_${dataset}_${model}_gc_${grad_clip}" arc4_2nd_order_run.sh "${dataset}" "${model}" "${RF_level}" "normal" "${name}" "${algorithm}" "${grad_clip}" "${record_saturation}" "${epochs}" "${input_res}"

else

#sbatch --nodes=1 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="${algorithm}_rf_${RF_level}_${dataset}_${model}_gc_${grad_clip}.err" --output="${algorithm}_rf_${RF_level}_${dataset}_${model}_gc_${grad_clip}.out"  --job-name="${algorithm}_rf_${RF_level}_${dataset}_${model}_gc_${grad_clip}" slurm_2nd_order_run.sh "${dataset}" "${model}" "${RF_level}" "normal" "${name}" "${algorithm}" "${grad_clip}" "${record_saturation}" "${epochs}" "${input_res}"
qsub -l h_rt=36:00:00 -l coproc_v100=1  -N  "${algorithm}_rf_${RF_level}_${dataset}_${model}_gc_${grad_clip}" arc4_2nd_order_run.sh "${dataset}" "${model}" "${RF_level}" "normal" "${name}" "${algorithm}" "${grad_clip}" "${record_saturation}" "${epochs}" "${input_res}"
fi
}

#grad_clip=1
#
#sbatch --nodes=1 --array 1-3 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="ekfac_optim_rf_3_cifar10_resnet50_gc_${grad_clip}.err" --output="ekfac_optim_rf_3_cifar10_resnet50_gc_${grad_clip}.output"  --job-name="ekfac_optim_rf_3_cifar10_resnet50_gc_${grad_clip}" slurm_2nd_order_run.sh "cifar10" "resnet50" "3" "normal" "ekfac_optim_hyper_saturation_200_gc_${grad_clip}" "1" grad_clip
#sbatch --nodes=1 --array 1-3 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="ekfac_optim_rf_4_cifar10_resnet50_gc_${grad_clip}.err" --output="ekfac_optim_rf_4_cifar10_resnet50_gc_${grad_clip}.output" --job-name="ekfac_optim_rf_4_cifar10_resnet50_gc_${grad_clip}" slurm_2nd_order_run.sh "cifar10" "resnet50" "4" "normal" "ekfac_optim_hyper_saturation_200_gc_${grad_clip}" "1" grad_clip

#sbatch --nodes=1 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="ekfac_optim_rf_1_cifar10_resnet50_gc_${grad_clip}.err" --output="ekfac_optim_rf_1_cifar10_resnet50_gc_${grad_clip}.output"  --job-name="ekfac_optim_rf_3_cifar10_resnet50_gc_${grad_clip}" slurm_2nd_order_run.sh "cifar10" "resnet50" "1" "normal" "ekfac_optim_hyper_saturation_200_gc_${grad_clip}" "1" "${grad_clip}" 1
#sbatch --nodes=1 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="ekfac_optim_rf_2_cifar10_resnet50_gc_${grad_clip}.err" --output="ekfac_optim_rf_2_cifar10_resnet50_gc_${grad_clip}.output"  --job-name="ekfac_optim_rf_3_cifar10_resnet50_gc_${grad_clip}" slurm_2nd_order_run.sh "cifar10" "resnet50" "2" "normal" "ekfac_optim_hyper_saturation_200_gc_${grad_clip}" "1" "${grad_clip}" 1

res=224
epochs=100
array=1
ffcv=1

if [ "${ffcv}" -eq 1 ]; then
  ffcv_string="ffcv"
  else
  ffcv_string="no_ffcv"
  fi
for model in "resnet25_small"; do
for dataset in "small_imagenet"; do
#for model in "deep_small_vgg" "resnet25_small"; do # all two models
#for lvl in 1 2 3 4; do                #
#for lvl in 5 6 7 8 10; do                # iterate idxA from 0 to length
for lvl in 1; do
# iterate idxA from 0 to length
if [ "${array}" -eq 1 ]; then

qsub -l h_rt=44:00:00 -t 1-3 -l coproc_v100=1 -N "deep_${model}_lvl_${lvl}_${dataset}_res_${res}" arc4_training_run.sh "${model}" "${dataset}" 8 ${lvl}  "normal" "${epochs}" "sgd_${epochs}_res_${res}_${ffcv_string}" 1 0 "${res}" "${ffcv}"

else

 qsub -l h_rt=36:00:00 -l coproc_v100=1 "deep_${model}_lvl_${lvl}_${dataset}_res_${res}"  arc4_training_run.sh "${model}" "${dataset}" 8 ${lvl}  "normal" "${epochs}" "sgd_${epochs}_res_${res}_${ffcv_sting}" 1 0 "${res}" "${ffcv}"
fi

#sbatch --nodes=1 --time=140:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="deep_{}_lvl_${lvl}_small_imagenet_res_${res}.err" --gres=gpu:1 --output="deep_small_resnet_lvl_${lvl}_small_imagenet_res_${res}.output"  --job-name="deep_small_resnet_lvl_${lvl}_small_imagenet_res_${res}" slurm_deep_small_resnet_run.sh "resnet25_small" "small_imagenet" 8 ${lvl}  "normal" "${epochs}" "recording_${epochs}" 1 1 "${res}"
#sbatch --nodes=1 --time=140:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="resnet_small_lvl_${lvl}_small_imagenet_res_${res}_saturation_${epochs}.err" --gres=gpu:1 --output="resnet_small_lvl_${lvl}_small_imagenet_res_${res}_saturation_${epochs}.output"  --job-name="resnet_small_lvl_${lvl}_small_imagenet_res_${res}_saturation_${epochs}" slurm_deep_small_resnet_run.sh "resnet_small" "small_imagenet" 4 ${lvl}  "normal" 100 "recording_100_res_${res}_ffcv" 1 1 "${res}" 1 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv"
done
done
done

