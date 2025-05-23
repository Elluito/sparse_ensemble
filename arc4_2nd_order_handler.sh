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

grad_clip=0
epochs=100
resolution=224
array=1
for model in "resnet25_small"; do
  for lvl in 5 6 7 8 10; do
#  for lvl in 5 6 7 8 10; do
    for optim in "sam"; do
      for dataset in "cifar10"; do

  #for model in "deep_small_vgg" "resnet25_small"; do # all two models

        run_2nd_order_experiment "${model}" "${dataset}" "${optim}" "${optim}_${dataset}_${epochs}_res_${resolution}_gc_${grad_clip}" "${epochs}" "${grad_clip}" "${lvl}" 0 "${array}" "${resolution}"

done
done
done
done

