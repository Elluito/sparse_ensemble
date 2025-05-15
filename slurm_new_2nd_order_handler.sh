#!/bin/bash
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

sbatch --nodes=1 --array=1-4 --time=48:00:00 --partition=gpu --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="${algorithm}_rf_${RF_level}_${dataset}_${model}_gc_${grad_clip}.err" --output="${algorithm}_rf_${RF_level}_${dataset}_${model}_gc_${grad_clip}.out"  --job-name="${algorithm}_rf_${RF_level}_${dataset}_${model}_gc_${grad_clipa}_${name}" slurm_new_2nd_order_run.sh "${dataset}" "${model}" "${RF_level}" "normal" "${name}" "${algorithm}" "${grad_clip}" "${record_saturation}" "${epochs}" "${input_res}"

else

sbatch --nodes=1 --time=48:00:00 --partition=gpu --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="${algorithm}_rf_${RF_level}_${dataset}_${model}_gc_${grad_clip}.err" --output="${algorithm}_rf_${RF_level}_${dataset}_${model}_gc_${grad_clip}.out"  --job-name="${algorithm}_rf_${RF_level}_${dataset}_${model}_gc_${grad_clip}_${name}" slurm_new_2nd_order_run.sh "${dataset}" "${model}" "${RF_level}" "normal" "${name}" "${algorithm}" "${grad_clip}" "${record_saturation}" "${epochs}" "${input_res}"

fi
}

#grad_clip=1
#
#sbatch --nodes=1 --array 1-3 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="ekfac_optim_rf_3_cifar10_resnet50_gc_${grad_clip}.err" --output="ekfac_optim_rf_3_cifar10_resnet50_gc_${grad_clip}.output"  --job-name="ekfac_optim_rf_3_cifar10_resnet50_gc_${grad_clip}" slurm_2nd_order_run.sh "cifar10" "resnet50" "3" "normal" "ekfac_optim_hyper_saturation_200_gc_${grad_clip}" "1" grad_clip
#sbatch --nodes=1 --array 1-3 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="ekfac_optim_rf_4_cifar10_resnet50_gc_${grad_clip}.err" --output="ekfac_optim_rf_4_cifar10_resnet50_gc_${grad_clip}.output" --job-name="ekfac_optim_rf_4_cifar10_resnet50_gc_${grad_clip}" slurm_2nd_order_run.sh "cifar10" "resnet50" "4" "normal" "ekfac_optim_hyper_saturation_200_gc_${grad_clip}" "1" grad_clip

#sbatch --nodes=1 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="ekfac_optim_rf_1_cifar10_resnet50_gc_${grad_clip}.err" --output="ekfac_optim_rf_1_cifar10_resnet50_gc_${grad_clip}.output"  --job-name="ekfac_optim_rf_3_cifar10_resnet50_gc_${grad_clip}" slurm_2nd_order_run.sh "cifar10" "resnet50" "1" "normal" "ekfac_optim_hyper_saturation_200_gc_${grad_clip}" "1" "${grad_clip}" 1
#sbatch --nodes=1 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="ekfac_optim_rf_2_cifar10_resnet50_gc_${grad_clip}.err" --output="ekfac_optim_rf_2_cifar10_resnet50_gc_${grad_clip}.output"  --job-name="ekfac_optim_rf_3_cifar10_resnet50_gc_${grad_clip}" slurm_2nd_order_run.sh "cifar10" "resnet50" "2" "normal" "ekfac_optim_hyper_saturation_200_gc_${grad_clip}" "1" "${grad_clip}" 1

grad_clip=0
epochs=200
resolution=32
array=1
for model in "resnet50" "vgg19"; do
#for model in "deep_small_vgg" "resnet25_small"; do # all two models
  for lvl in 3 4 ; do
#  for lvl in 5 6 7 8 10; do
    for optim in "ekfac"; do
      for dataset in "cifar10"; do


        run_2nd_order_experiment "${model}" "${dataset}" "${optim}" "lr_0.1_momentum_0.9_${optim}_${epochs}_res_${resolution}_gc_${grad_clip}" "${epochs}" "${grad_clip}" "${lvl}" 1 "${array}" "${resolution}"

done
done
done
done

