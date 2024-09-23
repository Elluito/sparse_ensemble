#!/bin/bash

#sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="ekfac_optim_rf_1_cifar10_vgg19_gc_${grad_clip}.err" --output="ekfac_optim_rf_1_cifar10_vgg19_gc_${grad_clip}.output"  --job-name="ekfac_optim_rf_1_cifar10_vgg19_gc_${grad_clip}" slurm_2nd_order_run.sh "cifar10" "vgg19" "1" "normal" "ekfac_optim_hyper_100_gc_${grad_clip}" "1" grad_clip
#sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="ekfac_optim_rf_2_cifar10_vgg19_gc_${grad_clip}.err" --output="ekfac_optim_rf_2_cifar10_vgg19_gc_${grad_clip}.output" --job-name="ekfac_optim_rf_2_cifar10_vgg19_gc_${grad_clip}" slurm_2nd_order_run.sh "cifar10" "vgg19" "2" "normal" "ekfac_optim_hyper_100_gc_${grad_clip}" "1" grad_clip
#sbatch --nodes=1 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="ekfac_optim_rf_3_cifar10_vgg19_gc_${grad_clip}.err" --output="ekfac_optim_rf_3_cifar10_vgg19_gc_${grad_clip}.output"  --job-name="ekfac_optim_rf_3_cifar10_vgg19_gc_${grad_clip}" slurm_2nd_order_run.sh "cifar10" "vgg19" "3" "normal" "ekfac_optim_hyper_saturation_200_gc_${grad_clip}" "1" grad_clip
#sbatch --nodes=1 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="ekfac_optim_rf_4_cifar10_vgg19_gc_${grad_clip}.err" --output="ekfac_optim_rf_4_cifar10_vgg19_gc_${grad_clip}.output" --job-name="ekfac_optim_rf_4_cifar10_vgg19_gc_${grad_clip}" slurm_2nd_order_run.sh "cifar10" "vgg19" "4" "normal" "ekfac_optim_hyper_saturation_200_gc_${grad_clip}" "1" grad_clip

#grad_clip=1
#
#sbatch --nodes=1 --array 1-3 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="ekfac_optim_rf_3_cifar10_resnet50_gc_${grad_clip}.err" --output="ekfac_optim_rf_3_cifar10_resnet50_gc_${grad_clip}.output"  --job-name="ekfac_optim_rf_3_cifar10_resnet50_gc_${grad_clip}" slurm_2nd_order_run.sh "cifar10" "resnet50" "3" "normal" "ekfac_optim_hyper_saturation_200_gc_${grad_clip}" "1" grad_clip
#sbatch --nodes=1 --array 1-3 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="ekfac_optim_rf_4_cifar10_resnet50_gc_${grad_clip}.err" --output="ekfac_optim_rf_4_cifar10_resnet50_gc_${grad_clip}.output" --job-name="ekfac_optim_rf_4_cifar10_resnet50_gc_${grad_clip}" slurm_2nd_order_run.sh "cifar10" "resnet50" "4" "normal" "ekfac_optim_hyper_saturation_200_gc_${grad_clip}" "1" grad_clip
grad_clip=0

#sbatch --nodes=1 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="ekfac_optim_rf_1_cifar10_resnet50_gc_${grad_clip}.err" --output="ekfac_optim_rf_1_cifar10_resnet50_gc_${grad_clip}.output"  --job-name="ekfac_optim_rf_3_cifar10_resnet50_gc_${grad_clip}" slurm_2nd_order_run.sh "cifar10" "resnet50" "1" "normal" "ekfac_optim_hyper_saturation_200_gc_${grad_clip}" "1" "${grad_clip}" 1
#sbatch --nodes=1 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="ekfac_optim_rf_2_cifar10_resnet50_gc_${grad_clip}.err" --output="ekfac_optim_rf_2_cifar10_resnet50_gc_${grad_clip}.output"  --job-name="ekfac_optim_rf_3_cifar10_resnet50_gc_${grad_clip}" slurm_2nd_order_run.sh "cifar10" "resnet50" "2" "normal" "ekfac_optim_hyper_saturation_200_gc_${grad_clip}" "1" "${grad_clip}" 1

###################################### EKFAC ##################################################################################

#sbatch --nodes=1 --array=1-3 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="ekfac_optim_rf_1_cifar10_resnet50_gc_${grad_clip}.err" --output="ekfac_optim_rf_1_cifar10_resnet50_gc_${grad_clip}.output"  --job-name="ekfac_optim_rf_1_cifar10_resnet50_gc_${grad_clip}" slurm_2nd_order_run.sh "cifar10" "resnet50" "1" "normal" "ekfac_optim_hyper_saturation_200_gc_${grad_clip}" "1" "${grad_clip}" 1
#sbatch --nodes=1 --array=1-3 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="ekfac_optim_rf_2_cifar10_resnet50_gc_${grad_clip}.err" --output="ekfac_optim_rf_2_cifar10_resnet50_gc_${grad_clip}.output"  --job-name="ekfac_optim_rf_2_cifar10_resnet50_gc_${grad_clip}" slurm_2nd_order_run.sh "cifar10" "resnet50" "2" "normal" "ekfac_optim_hyper_saturation_200_gc_${grad_clip}" "1" "${grad_clip}" 1
#
#
#
#sbatch --nodes=1 --array=1-3 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="ekfac_optim_rf_3_cifar10_resnet50_gc_${grad_clip}.err" --output="ekfac_optim_rf_3_cifar10_resnet50_gc_${grad_clip}.output"  --job-name="ekfac_optim_rf_3_cifar10_resnet50_gc_${grad_clip}" slurm_2nd_order_run.sh "cifar10" "resnet50" "3" "normal" "ekfac_optim_hyper_saturation_200_gc_${grad_clip}" "1" "${grad_clip}" 1
#sbatch --nodes=1 --array=1-3 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="ekfac_optim_rf_4_cifar10_resnet50_gc_${grad_clip}.err" --output="ekfac_optim_rf_4_cifar10_resnet50_gc_${grad_clip}.output" --job-name="ekfac_optim_rf_4_cifar10_resnet50_gc_${grad_clip}" slurm_2nd_order_run.sh "cifar10" "resnet50" "4" "normal" "ekfac_optim_hyper_saturation_200_gc_${grad_clip}" "1" "${grad_clip}" 1
#
#
#
#sbatch --nodes=1 --array=1-3 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="ekfac_optim_rf_1_cifar10_vgg19_gc_${grad_clip}.err" --output="ekfac_optim_rf_1_cifar10_vgg19_gc_${grad_clip}.output"  --job-name="ekfac_optim_rf_1_cifar10_vgg19_gc_${grad_clip}" slurm_2nd_order_run.sh "cifar10" "vgg19" "1" "normal" "ekfac_optim_hyper_saturation_200_gc_${grad_clip}" "1" "${grad_clip}" 1
#sbatch --nodes=1 --array=1-3 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="ekfac_optim_rf_2_cifar10_vgg19_50_gc_${grad_clip}.err" --output="ekfac_optim_rf_2_cifar10_vgg19_gc_${grad_clip}.output"  --job-name="ekfac_optim_rf_2_cifar10_vgg19_gc_${grad_clip}" slurm_2nd_order_run.sh "cifar10" "vgg19" "2" "normal" "ekfac_optim_hyper_saturation_200_gc_${grad_clip}" "1" "${grad_clip}" 1
#
#
#sbatch --nodes=1 --array=1-3 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="ekfac_optim_rf_3_cifar10_vgg19_gc_${grad_clip}.err" --output="ekfac_optim_rf_3_cifar10_vgg19_gc_${grad_clip}.output"  --job-name="ekfac_optim_rf_3_cifar10_vgg19_gc_${grad_clip}" slurm_2nd_order_run.sh "cifar10" "vgg19" "3" "normal" "ekfac_optim_hyper_saturation_200_gc_${grad_clip}" "1" "${grad_clip}" 1
#sbatch --nodes=1 --array=1-3 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="ekfac_optim_rf_4_cifar10_vgg19_gc_${grad_clip}.err" --output="ekfac_optim_rf_4_cifar10_vgg19_gc_${grad_clip}.output" --job-name="ekfac_optim_rf_4_cifar10_vgg19_gc_${grad_clip}" slurm_2nd_order_run.sh "cifar10" "vgg19" "4" "normal" "ekfac_optim_hyper_saturation_200_gc_${grad_clip}" "1" "${grad_clip}" 1

###################################### SAM ##################################################################################

#sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="sam_optim_rf_1_cifar10_rs.err" --output="sam_optim_rf_1_cifar10_rs.output" --job-name="sam_optim_rf_1_cifar10_rs" slurm_2nd_order_run.sh "cifar10" "resnet50" "1" "normal" "sam_optim_hyper_200" "2"
#sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="sam_optim_rf_2_cifar10_rs.err" --output="sam_optim_rf_2_cifar10_rs.output" --job-name="sam_optim_rf_2_cifar10_rs" slurm_2nd_order_run.sh "cifar10" "resnet50" "2" "normal" "sam_optim_hyper_200" "2"
grad_clip=0


#sbatch --nodes=1 --array=1-3 --time=50:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="sam_optim_rf_1_cifar10_rs.err" --output="sam_optim_rf_1_cifar10_rs.output" --job-name="sam_optim_rf_1_cifar10_rs" slurm_2nd_order_run.sh "cifar10" "resnet50" "1" "normal" "sam_optim_saturation_200_gc_${grad_clip}" "2" "${grad_clip}" 1
#sbatch --nodes=1 --array=1-3 --time=50:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="sam_optim_rf_2_cifar10_rs.err" --output="sam_optim_rf_2_cifar10_rs.output" --job-name="sam_optim_rf_2_cifar10_rs" slurm_2nd_order_run.sh "cifar10" "resnet50" "2" "normal" "sam_optim_saturation_200_gc_${grad_clip}" "2" "${grad_clip}" 1




#sbatch --nodes=1 --array=1-2 --time=50:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="sam_optim_rf_3_cifar10_rs.err" --output="sam_optim_rf_3_cifar10_rs.output" --job-name="sam_optim_rf_3_cifar10_rs" slurm_2nd_order_run.sh "cifar10" "resnet50" "3" "normal" "sam_optim_saturation_200_gc_${grad_clip}" "2" "${grad_clip}" 1
#sbatch --nodes=1 --array=1-3 --time=50:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="sam_optim_rf_4_cifar10_rs.err" --output="sam_optim_rf_4_cifar10_rs.output" --job-name="sam_optim_rf_4_cifar10_rs" slurm_2nd_order_run.sh "cifar10" "resnet50" "4" "normal" "sam_optim_saturation_200_gc_${grad_clip}" "2" "${grad_clip}" 1


#
#sbatch --nodes=1 --array=1-3 --time=50:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="sam_optim_rf_1_cifar10_vgg19.err" --output="sam_optim_rf_1_cifar10_vgg19.output" --job-name="sam_optim_rf_1_cifar10_vgg19" slurm_2nd_order_run.sh "cifar10" "vgg19" "1" "normal" "sam_optim_saturation_200_gc_${grad_clip}" "2" "${grad_clip}" 1
#sbatch --nodes=1  --array=1-3 --time=50:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="sam_optim_rf_2_cifar10_vgg19.err" --output="sam_optim_rf_2_cifar10_vgg19.output" --job-name="sam_optim_rf_2_cifar10_vgg19" slurm_2nd_order_run.sh "cifar10" "vgg19" "2" "normal" "sam_optim_saturation_200_gc_${grad_clip}" "2" "${grad_clip}" 1
#
#sbatch --nodes=1 --array=1-3 --time=50:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="sam_optim_rf_3_cifar10_vgg19.err" --output="sam_optim_rf_3_cifar10_vgg19.output" --job-name="sam_optim_rf_3_cifar10_vgg19" slurm_2nd_order_run.sh "cifar10" "vgg19" "3" "normal" "sam_optim_saturation_200_gc_${grad_clip}" "2" "${grad_clip}" 1
#sbatch --nodes=1  --array=1-3 --time=50:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="sam_optim_rf_4_cifar10_vgg19.err" --output="sam_optim_rf_4_cifar10_vgg19.output" --job-name="sam_optim_rf_4_cifar10_vgg19" slurm_2nd_order_run.sh "cifar10" "vgg19" "4" "normal" "sam_optim_saturation_200_gc_${grad_clip}" "2" "${grad_clip}" 1





#sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="sam_optim_rf_5_cifar10_rs.err" --output="sam_optim_rf_5_cifar10_rs.output" --job-name="sam_optim_rf_5_cifar10_rs" slurm_2nd_order_run.sh "cifar10" "resnet50" "5" "normal" "sam_optim_hyper_200" "2"
#sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="sam_optim_rf_6_cifar10_rs.err" --output="sam_optim_rf_6_cifar10_rs.output" --job-name="sam_optim_rf_6_cifar10_rs" slurm_2nd_order_run.sh "cifar10" "resnet50" "6" "normal" "sam_optim_hyper_200" "2"
#sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="sam_optim_rf_7_cifar10_rs.err" --output="sam_optim_rf_7_cifar10_rs.output" --job-name="sam_optim_rf_7_cifar10_rs" slurm_2nd_order_run.sh "cifar10" "resnet50" "7" "normal" "sam_optim_hyper_200" "2"
#sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="sam_optim_rf_8_cifar10_rs.err" --output="sam_optim_rf_8_cifar10_rs.output"  --job-name="sam_optim_rf_8_cifar10_rs" slurm_2nd_order_run.sh "cifar10" "resnet50" "8" "normal" "sam_optim_hyper_200" "2"
