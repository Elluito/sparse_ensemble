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

sbatch --nodes=1 --array=1-4 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="${algorithm}_rf_${RF_level}_${dataset}_${model}_gc_${grad_clip}.err" --output="${algorithm}_rf_${RF_level}_${dataset}_${model}_gc_${grad_clip}.out"  --job-name="${algorithm}_rf_${RF_level}_${dataset}_${model}_gc_${grad_clip}" slurm_2nd_order_run.sh "${dataset}" "${model}" "${RF_level}" "normal" "${name}" "${algorithm}" "${grad_clip}" "${record_saturation}" "${epochs}" "${input_res}"

else

sbatch --nodes=1 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="${algorithm}_rf_${RF_level}_${dataset}_${model}_gc_${grad_clip}.err" --output="${algorithm}_rf_${RF_level}_${dataset}_${model}_gc_${grad_clip}.out"  --job-name="${algorithm}_rf_${RF_level}_${dataset}_${model}_gc_${grad_clip}" slurm_2nd_order_run.sh "${dataset}" "${model}" "${RF_level}" "normal" "${name}" "${algorithm}" "${grad_clip}" "${record_saturation}" "${epochs}" "${input_res}"

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
array=0
for model in "resnet25_small"; do
#for model in "deep_small_vgg" "resnet25_small"; do # all two models
  for lvl in 6 7; do
    for optim in "sam"; do
      for dataset in "cifar10"; do


        run_2nd_order_experiment "${model}" "${dataset}" "${optim}" "${optim}_${dataset}_${epochs}_res_${resolution}_gc_${grad_clip}" "${epochs}" "${grad_clip}" "${lvl}" 1 "${array}" "${resolution}"

done
done
done
done

###################################### EKFAC ##################################################################################

#sbatch --nodes=1 --array=1-3 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="ekfac_optim_rf_1_cifar10_resnet50_gc_${grad_clip}.err" --output="ekfac_optim_rf_1_cifar10_resnet50_gc_${grad_clip}.output"  --job-name="ekfac_optim_rf_1_cifar10_resnet50_gc_${grad_clip}" slurm_2nd_order_run.sh "cifar10" "resnet50" "1" "normal" "ekfac_optim_hyper_saturation_200_gc_${grad_clip}" "1" "${grad_clip}" 1
#sbatch --nodes=1 --array=1-3 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="ekfac_optim_rf_2_cifar10_resnet50_gc_${grad_clip}.err" --output="ekfac_optim_rf_2_cifar10_resnet50_gc_${grad_clip}.output"  --job-name="ekfac_optim_rf_2_cifar10_resnet50_gc_${grad_clip}" slurm_2nd_order_run.sh "cifar10" "resnet50" "2" "normal" "ekfac_optim_hyper_saturation_200_gc_${grad_clip}" "1" "${grad_clip}" 1
#sbatch --nodes=1 --array=1-3 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="ekfac_optim_rf_3_cifar10_resnet50_gc_${grad_clip}.err" --output="ekfac_optim_rf_3_cifar10_resnet50_gc_${grad_clip}.output"  --job-name="ekfac_optim_rf_3_cifar10_resnet50_gc_${grad_clip}" slurm_2nd_order_run.sh "cifar10" "resnet50" "3" "normal" "ekfac_optim_hyper_saturation_200_gc_${grad_clip}" "1" "${grad_clip}" 1
#sbatch --nodes=1 --array=1-3 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="ekfac_optim_rf_4_cifar10_resnet50_gc_${grad_clip}.err" --output="ekfac_optim_rf_4_cifar10_resnet50_gc_${grad_clip}.output" --job-name="ekfac_optim_rf_4_cifar10_resnet50_gc_${grad_clip}" slurm_2nd_order_run.sh "cifar10" "resnet50" "4" "normal" "ekfac_optim_hyper_saturation_200_gc_${grad_clip}" "1" "${grad_clip}" 1

#sbatch --nodes=1 --array=1-3 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="ekfac_optim_rf_5_cifar10_resnet50_gc_${grad_clip}.err" --output="ekfac_optim_rf_5_cifar10_resnet50_gc_${grad_clip}.output"  --job-name="ekfac_optim_rf_5_cifar10_resnet50_gc_${grad_clip}" slurm_2nd_order_run.sh "cifar10" "resnet50" "5" "normal" "ekfac_optim_hyper_saturation_200_gc_${grad_clip}" "1" "${grad_clip}" 1
#sbatch --nodes=1 --array=1-3 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="ekfac_optim_rf_6_cifar10_resnet50_gc_${grad_clip}.err" --output="ekfac_optim_rf_6_cifar10_resnet50_gc_${grad_clip}.output"  --job-name="ekfac_optim_rf_6_cifar10_resnet50_gc_${grad_clip}" slurm_2nd_order_run.sh "cifar10" "resnet50" "6" "normal" "ekfac_optim_hyper_saturation_200_gc_${grad_clip}" "1" "${grad_clip}" 1
#sbatch --nodes=1 --array=1-3 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="ekfac_optim_rf_7_cifar10_resnet50_gc_${grad_clip}.err" --output="ekfac_optim_rf_7_cifar10_resnet50_gc_${grad_clip}.output"  --job-name="ekfac_optim_rf_7_cifar10_resnet50_gc_${grad_clip}" slurm_2nd_order_run.sh "cifar10" "resnet50" "7" "normal" "ekfac_optim_hyper_saturation_200_gc_${grad_clip}" "1" "${grad_clip}" 1



#sbatch --nodes=1 --array=1-3 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="ekfac_optim_rf_1_cifar10_vgg19_gc_${grad_clip}.err" --output="ekfac_optim_rf_1_cifar10_vgg19_gc_${grad_clip}.output"  --job-name="ekfac_optim_rf_1_cifar10_vgg19_gc_${grad_clip}" slurm_2nd_order_run.sh "cifar10" "vgg19" "1" "normal" "ekfac_optim_hyper_saturation_200_gc_${grad_clip}" "1" "${grad_clip}" 1
#sbatch --nodes=1 --array=1-3 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="ekfac_optim_rf_2_cifar10_vgg19_50_gc_${grad_clip}.err" --output="ekfac_optim_rf_2_cifar10_vgg19_gc_${grad_clip}.output"  --job-name="ekfac_optim_rf_2_cifar10_vgg19_gc_${grad_clip}" slurm_2nd_order_run.sh "cifar10" "vgg19" "2" "normal" "ekfac_optim_hyper_saturation_200_gc_${grad_clip}" "1" "${grad_clip}" 1


#sbatch --nodes=1 --array=1-3 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="ekfac_optim_rf_3_cifar10_vgg19_gc_${grad_clip}.err" --output="ekfac_optim_rf_3_cifar10_vgg19_gc_${grad_clip}.output"  --job-name="ekfac_optim_rf_3_cifar10_vgg19_gc_${grad_clip}" slurm_2nd_order_run.sh "cifar10" "vgg19" "3" "normal" "ekfac_optim_hyper_saturation_200_gc_${grad_clip}" "1" "${grad_clip}" 1
#sbatch --nodes=1 --array=1-3 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="ekfac_optim_rf_4_cifar10_vgg19_gc_${grad_clip}.err" --output="ekfac_optim_rf_4_cifar10_vgg19_gc_${grad_clip}.output" --job-name="ekfac_optim_rf_4_cifar10_vgg19_gc_${grad_clip}" slurm_2nd_order_run.sh "cifar10" "vgg19" "4" "normal" "ekfac_optim_hyper_saturation_200_gc_${grad_clip}" "1" "${grad_clip}" 1

###################################### ASAM ##################################################################################

#sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="sam_optim_rf_1_cifar10_rs.err" --output="sam_optim_rf_1_cifar10_rs.output" --job-name="sam_optim_rf_1_cifar10_rs" slurm_2nd_order_run.sh "cifar10" "resnet50" "1" "normal" "sam_optim_hyper_200" "2"
#sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="sam_optim_rf_2_cifar10_rs.err" --output="sam_optim_rf_2_cifar10_rs.output" --job-name="sam_optim_rf_2_cifar10_rs" slurm_2nd_order_run.sh "cifar10" "resnet50" "2" "normal" "sam_optim_hyper_200" "2"
grad_clip=0


#sbatch --nodes=1 --array=1-3 --time=50:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="sam_optim_rf_1_cifar10_rs.err" --output="sam_optim_rf_1_cifar10_rs.output" --job-name="sam_optim_rf_1_cifar10_rs" slurm_2nd_order_run.sh "cifar10" "resnet50" "1" "normal" "sam_optim_saturation_200_gc_${grad_clip}" "2" "${grad_clip}" 1
#sbatch --nodes=1 --array=1-3 --time=50:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="sam_optim_rf_2_cifar10_rs.err" --output="sam_optim_rf_2_cifar10_rs.output" --job-name="sam_optim_rf_2_cifar10_rs" slurm_2nd_order_run.sh "cifar10" "resnet50" "2" "normal" "sam_optim_saturation_200_gc_${grad_clip}" "2" "${grad_clip}" 1




#sbatch --nodes=1 --array=1-2 --time=50:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="sam_optim_rf_3_cifar10_rs.err" --output="sam_optim_rf_3_cifar10_rs.output" --job-name="sam_optim_rf_3_cifar10_rs" slurm_2nd_order_run.sh "cifar10" "resnet50" "3" "normal" "sam_optim_saturation_200_gc_${grad_clip}" "2" "${grad_clip}" 1
#sbatch --nodes=1 --array=1-3 --time=50:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="sam_optim_rf_4_cifar10_rs.err" --output="sam_optim_rf_4_cifar10_rs.output" --job-name="sam_optim_rf_4_cifar10_rs" slurm_2nd_order_run.sh "cifar10" "resnet50" "4" "normal" "sam_optim_saturation_200_gc_${grad_clip}" "2" "${grad_clip}" 1
#
#
##
#sbatch --nodes=1 --array=1-2 --time=50:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="sam_optim_rf_1_cifar10_vgg19.err" --output="sam_optim_rf_1_cifar10_vgg19.output" --job-name="sam_optim_rf_1_cifar10_vgg19" slurm_2nd_order_run.sh "cifar10" "vgg19" "1" "normal" "sam_optim_saturation_200_gc_${grad_clip}" "2" "${grad_clip}" 1
#sbatch --nodes=1  --array=1-3 --time=50:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="sam_optim_rf_2_cifar10_vgg19.err" --output="sam_optim_rf_2_cifar10_vgg19.output" --job-name="sam_optim_rf_2_cifar10_vgg19" slurm_2nd_order_run.sh "cifar10" "vgg19" "2" "normal" "sam_optim_saturation_200_gc_${grad_clip}" "2" "${grad_clip}" 1
#
#sbatch --nodes=1 --array=1-3 --time=50:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="sam_optim_rf_3_cifar10_vgg19.err" --output="sam_optim_rf_3_cifar10_vgg19.output" --job-name="sam_optim_rf_3_cifar10_vgg19" slurm_2nd_order_run.sh "cifar10" "vgg19" "3" "normal" "sam_optim_saturation_200_gc_${grad_clip}" "2" "${grad_clip}" 1
#sbatch --nodes=1  --array=1-3 --time=50:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="sam_optim_rf_4_cifar10_vgg19.err" --output="sam_optim_rf_4_cifar10_vgg19.output" --job-name="sam_optim_rf_4_cifar10_vgg19" slurm_2nd_order_run.sh "cifar10" "vgg19" "4" "normal" "sam_optim_saturation_200_gc_${grad_clip}" "2" "${grad_clip}" 1





#sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="sam_optim_rf_5_cifar10_rs.err" --output="sam_optim_rf_5_cifar10_rs.output" --job-name="sam_optim_rf_5_cifar10_rs" slurm_2nd_order_run.sh "cifar10" "resnet50" "5" "normal" "sam_optim_hyper_200" "2"
#sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="sam_optim_rf_6_cifar10_rs.err" --output="sam_optim_rf_6_cifar10_rs.output" --job-name="sam_optim_rf_6_cifar10_rs" slurm_2nd_order_run.sh "cifar10" "resnet50" "6" "normal" "sam_optim_hyper_200" "2"
#sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="sam_optim_rf_7_cifar10_rs.err" --output="sam_optim_rf_7_cifar10_rs.output" --job-name="sam_optim_rf_7_cifar10_rs" slurm_2nd_order_run.sh "cifar10" "resnet50" "7" "normal" "sam_optim_hyper_200" "2"
#sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="sam_optim_rf_8_cifar10_rs.err" --output="sam_optim_rf_8_cifar10_rs.output"  --job-name="sam_optim_rf_8_cifar10_rs" slurm_2nd_order_run.sh "cifar10" "resnet50" "8" "normal" "sam_optim_hyper_200" "2"

