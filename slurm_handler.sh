#!/bin/bash
#
grad_clip=0.4
#sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="ekfac_optim_rf_1_cifar10_rs_gc_${grad_clip}.err" --output="ekfac_optim_rf_1_cifar10_rs_gc_${grad_clip}.output"  --job-name="ekfac_optim_rf_1_cifar10_rs_gc_${grad_clip}" slurm_2nd_order.sh "cifar10" "resnet50" "1" "normal" "ekfac_optim_hyper_100_gc_${grad_clip}" "1" grad_clip
#sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="ekfac_optim_rf_2_cifar10_rs_gc_${grad_clip}.err" --output="ekfac_optim_rf_2_cifar10_rs_gc_${grad_clip}.output" --job-name="ekfac_optim_rf_2_cifar10_rs_gc_${grad_clip}" slurm_2nd_order.sh "cifar10" "resnet50" "2" "normal" "ekfac_optim_hyper_100_gc_${grad_clip}" "1" grad_clip
#sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="ekfac_optim_rf_3_cifar10_rs_gc_${grad_clip}.err" --output="ekfac_optim_rf_3_cifar10_rs_gc_${grad_clip}.output"  --job-name="ekfac_optim_rf_3_cifar10_rs_gc_${grad_clip}" slurm_2nd_order.sh "cifar10" "resnet50" "3" "normal" "ekfac_optim_hyper_100_gc_${grad_clip}" "1" grad_clip
#sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="ekfac_optim_rf_4_cifar10_rs_gc_${grad_clip}.err" --output="ekfac_optim_rf_4_cifar10_rs_gc_${grad_clip}.output" --job-name="ekfac_optim_rf_4_cifar10_rs_gc_${grad_clip}" slurm_2nd_order.sh "cifar10" "resnet50" "4" "normal" "ekfac_optim_hyper_100_gc_${grad_clip}" "1" grad_clip
#sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="ekfac_optim_rf_5_cifar10_rs_gc_${grad_clip}.err" --output="ekfac_optim_rf_5_cifar10_rs_gc_${grad_clip}.output" --job-name="ekfac_optim_rf_5_cifar10_rs_gc_${grad_clip}" slurm_2nd_order.sh "cifar10" "resnet50" "5" "normal" "ekfac_optim_hyper_100_gc_${grad_clip}" "1" grad_clip
#sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="ekfac_optim_rf_6_cifar10_rs_gc_${grad_clip}.err" --output="ekfac_optim_rf_6_cifar10_rs_gc_${grad_clip}.output" --job-name="ekfac_optim_rf_6_cifar10_rs_gc_${grad_clip}" slurm_2nd_order.sh "cifar10" "resnet50" "6" "normal" "ekfac_optim_hyper_100_gc_${grad_clip}" "1" grad_clip
#sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="ekfac_optim_rf_7_cifar10_rs_gc_${grad_clip}.err" --output="ekfac_optim_rf_7_cifar10_rs_gc_${grad_clip}.output" --job-name="ekfac_optim_rf_7_cifar10_rs_gc_${grad_clip}" slurm_2nd_order.sh "cifar10" "resnet50" "7" "normal" "ekfac_optim_hyper_100_gc_${grad_clip}" "1" grad_clip
#sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="ekfac_optim_rf_8_cifar10_rs_gc_${grad_clip}.err" --output="ekfac_optim_rf_8_cifar10_rs_gc_${grad_clip}.output" --job-name="ekfac_optim_rf_8_cifar10_rs_gc_${grad_clip}" slurm_2nd_order.sh "cifar10" "resnet50" "8" "normal" "ekfac_optim_hyper_100_gc_${grad_clip}" "1" grad_clip
##
##
##
##
#sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="sam_optim_rf_1_cifar10_rs.err" --output="sam_optim_rf_1_cifar10_rs.output" --job-name="sam_optim_rf_1_cifar10_rs" slurm_2nd_order.sh "cifar10" "resnet50" "1" "normal" "sam_optim_hyper_200" "2"
#sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="sam_optim_rf_2_cifar10_rs.err" --output="sam_optim_rf_2_cifar10_rs.output" --job-name="sam_optim_rf_2_cifar10_rs" slurm_2nd_order.sh "cifar10" "resnet50" "2" "normal" "sam_optim_hyper_200" "2"
#sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="sam_optim_rf_3_cifar10_rs.err" --output="sam_optim_rf_3_cifar10_rs.output" --job-name="sam_optim_rf_3_cifar10_rs" slurm_2nd_order.sh "cifar10" "resnet50" "3" "normal" "sam_optim_hyper_200" "2"
#sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="sam_optim_rf_4_cifar10_rs.err" --output="sam_optim_rf_4_cifar10_rs.output" --job-name="sam_optim_rf_4_cifar10_rs" slurm_2nd_order.sh "cifar10" "resnet50" "4" "normal" "sam_optim_hyper_200" "2"
#sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="sam_optim_rf_k6_cifar10_rs.err" --output="sam_optim_rf_k6_cifar10_rs.output" --job-name="sam_optim_rf_k6_cifar10_rs" slurm_2nd_order.sh "cifar10" "resnet50" "k6" "normal" "sam_optim_hyper_200" "2"
#sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="sam_optim_rf_k7_cifar10_rs.err" --output="sam_optim_rf_k7_cifar10_rs.output" --job-name="sam_optim_rf_k7_cifar10_rs" slurm_2nd_order.sh "cifar10" "resnet50" "k7" "normal" "sam_optim_hyper_200" "2"
#sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="sam_optim_rf_k8_cifar10_rs.err" --output="sam_optim_rf_k8_cifar10_rs.output" --job-name="sam_optim_rf_k8_cifar10_rs" slurm_2nd_order.sh "cifar10" "resnet50" "k8" "normal" "sam_optim_hyper_200" "2"
#sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="sam_optim_rf_k9_cifar10_rs.err" --output="sam_optim_rf_k9_cifar10_rs.output"  --job-name="sam_optim_rf_k9_cifar10_rs" slurm_2nd_order.sh "cifar10" "resnet50" "k9" "normal" "sam_optim_hyper_200" "2"
#
#sbatch --nodes=1 --time=00:05:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="import_test.err" --gres=gpu:1 --output="import_test.output"  --job-name="import_test" slurm_run.sh

#sbatch --nodes=1 --time=03:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="pruning_imagenet_resnet34.err" --gres=gpu:1 --output="pruning_imagenet_resnet34.output"  --job-name="pruning_imagenet_resnet34" slurm_run.sh "resnet34"
#sbatch --nodes=1 --time=03:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="pruning_imagenet_legacy_seresnet34.err" --gres=gpu:1 --output="pruning_imagenet_legacy_seresnet34.output"  --job-name="pruning_imagenet_legacy_seresnet34" slurm_run.sh "legacy_seresnet34"
#sbatch --nodes=1 --time=03:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="pruning_imagenet_skresnet34.err" --gres=gpu:1 --output="pruning_imagenet_skresnet34.output"  --job-name="pruning_imagenet_skresnet34" slurm_run.sh "skresnet34"
#sbatch --nodes=1 --time=03:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="pruning_imagenet_mobilenetv2.err" --gres=gpu:1 --output="pruning_imagenet_mobilenetv2.output"  --job-name="pruning_imagenet_mobilenetv2" slurm_run.sh "mobilenetv2"
#sbatch --nodes=1 --time=03:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="pruning_imagenet_mobilenetv3.err" --gres=gpu:1 --output="pruning_imagenet_mobilenetv3.output"  --job-name="pruning_imagenet_mobilenetv3" slurm_run.sh "mobilenetv3"
#sbatch --nodes=1 --time=03:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="pruning_imagenet_efficientnet.err" --gres=gpu:1 --output="pruning_imagenet_efficientnet.output"  --job-name="pruning_imagenet_efficientnet" slurm_run.sh "efficientnet"

#"resnet_small" "small_imagenet" 2 3 "normal" 200 "recording_200" 1 1

#sbatch --nodes=1 --time=96:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="vgg19_lv_2_small_imagenet_ffcv.err" --gres=gpu:1 --output="vgg_lv_2_small_imagenet_ffcv.output"  --job-name="vgg_lv_2_small_imagenet_ffcv" slurm_run_ffcv.sh "vgg19" "small_imagenet" 2 2 "normal" 200 "recording_200" 1 1

#########################     small resnet         Small imagenet    ####################################

#sbatch --nodes=1 --time=96:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="small_resnet_lv_8_small_imagenet.err" --gres=gpu:1 --output="small_resnet_lv_8_small_imagenet.output"  --job-name="small_resnet_lv_8_small_imagenet" slurm_run.sh "resnet_small" "small_imagenet" 1 8 "normal" 200 "recording_200" 1 1
#sbatch --nodes=1 --time=96:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="small_resnet_lv_9_small_imagenet.err" --gres=gpu:1 --output="small_resnet_lv_9_small_imagenet.output"  --job-name="small_resnet_lv_9_small_imagenet" slurm_run.sh "resnet_small" "small_imagenet" 2 9 "normal" 200 "recording_200" 1 1
#sbatch --nodes=1 --time=96:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="small_resnet_lv_10_small_imagenet.err" --gres=gpu:1 --output="small_resnet_lv_10_small_imagenet.output"  --job-name="small_resnet_lv_10_small_imagenet" slurm_run.sh "resnet_small" "small_imagenet" 1 10 "normal" 200 "recording_200" 1 1

##################################################   Not FFCV variant array ##################################################


#sbatch --nodes=1 --time=140:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="test_small_resnet_lv_5_small_imagenet_no_ffcv.err" --gres=gpu:1 --output="test_small_resnet_lv_5_small_imagenet_no_ffcv.output"  --job-name="test_small_resnet_lv_5_small_imagenet_no_ffcv" slurm_run.sh "resnet_small" "small_imagenet" 4 5 "normal" 200 "test_no_ffcv" 1 0

#sbatch --array=1-4 --nodes=1 --time=140:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="task_small_resnet_lv_3_small_imagenet_ffcv.err" --gres=gpu:1 --output="task_small_resnet_lv_3_small_imagenet_ffcv.output"  --job-name="task_small_resnet_lv_3_small_imagenet_ffcv" slurm_run_ffcv.sh "resnet_small" "small_imagenet" 4 3 "normal" 200 "recording_200_ffcv" 1 1 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv"
#sbatch --array=1-4 --nodes=1 --time=140:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="small_resnet_lv_4_small_imagenet_no_ffcv.err" --gres=gpu:1 --output="small_resnet_lv_4_small_imagenet_no_ffcv.output"  --job-name="small_resnet_lv_4_small_imagenet_no_ffcv" slurm_run.sh "resnet_small" "small_imagenet" 4 4 "normal" 200 "recording_200_no_ffcv" 1 1 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv"
#sbatch --nodes=1 --time=140:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="small_resnet_lv_5_small_imagenet_no_ffcv.err" --gres=gpu:1 --output="small_resnet_lv_5_small_imagenet_no_ffcv.output"  --job-name="small_resnet_lv_5_small_imagenet_no_ffcv" slurm_run.sh "resnet_small" "small_imagenet" 4 5 "normal" 200 "recording_200_no_ffcv" 1 1 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv"
#sbatch --array=1-4 --nodes=1 --time=140:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="task_small_resnet_lv_6_small_imagenet_ffcv.err" --gres=gpu:1 --output="task_small_resnet_lv_6_small_imagenet_ffcv.output"  --job-name="task_small_resnet_lv_6_small_imagenet_ffcv" slurm_run_ffcv.sh "resnet_small" "small_imagenet" 4 6 "normal" 200 "recording_200_ffcv" 1 1 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv"
#sbatch --nodes=1 --time=140:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="small_resnet_lv_7_small_imagenet_no_ffcv.err" --gres=gpu:1 --output="small_resnet_lv_7_small_imagenet_no_ffcv.output"  --job-name="small_resnet_lv_7_small_imagenet_no_ffcv" slurm_run.sh "resnet_small" "small_imagenet" 4 7 "normal" 200 "recording_200_no_ffcv" 1 1 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv"
#sbatch --nodes=1 --time=140:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="small_resnet_lv_8_small_imagenet_no_ffcv.err" --gres=gpu:1 --output="small_resnet_lv_8_small_imagenet_no_ffcv.output"  --job-name="small_resnet_lv_8_small_imagenet_no_ffcv" slurm_run.sh "resnet_small" "small_imagenet" 4 8 "normal" 200 "recording_200_no_ffcv" 1 1 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv"
#sbatch --nodes=1 --time=140:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="small_resnet_lv_9_small_imagenet_no_ffcv.err" --gres=gpu:1 --output="small_resnet_lv_9_small_imagenet_no_ffcv.output"  --job-name="small_resnet_lv_9_small_imagenet_no_ffcv" slurm_run.sh "resnet_small" "small_imagenet" 4 9 "normal" 200 "recording_200_no_ffcv" 1 1 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv"

#                                                   FFCV variants

#sbatch --array=1-4 --nodes=1 --time=140:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="task_small_resnet_lv_3_small_imagenet_ffcv.err" --gres=gpu:1 --output="task_small_resnet_lv_3_small_imagenet_no_ffcv.output"  --job-name="task_small_resnet_lv_3_small_imagenet_no_ffcv" slurm_run_ffcv.sh "resnet_small" "small_imagenet" 4 3 "normal" 200 "recording_200_no_ffcv" 1 1 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv"
#sbatch --array=1-4 --nodes=1 --time=140:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="task_small_resnet_lv_4_small_imagenet_ffcv.err" --gres=gpu:1 --output="task_small_resnet_lv_4_small_imagenet_no_ffcv.output"  --job-name="task_small_resnet_lv_4_small_imagenet_no_ffcv" slurm_run_ffcv.sh "resnet_small" "small_imagenet" 4 4 "normal" 200 "recording_200_no_ffcv" 1 1 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv"
#sbatch --array=1-4 --nodes=1 --time=140:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="task_small_resnet_lv_10_small_imagenet_ffcv.err" --gres=gpu:1 --output="task_small_resnet_lv_10_small_imagenet_no_ffcv.output"  --job-name="task_small_resnet_lv_10_small_imagenet_no_ffcv" slurm_run_ffcv.sh "resnet_small" "small_imagenet" 4 10 "normal" 200 "recording_200_no_ffcv" 1 1 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv"


############################## Resume small resnet #######################################################
#qsub -l h_rt=48:00:00 -l coproc_v100=1 -N "resume_training_Level_3_small_resnet_small_imagenet" resume_run.sh "resnet_small" "small_imagenet" 2 3 "normal" 200 "recording_200" 1 1 "/nobackup/sclaam/checkpoints/resnet_small_normal_small_imagenet_seed.0_rf_level_3_recording_200_test_acc_42.89.pth"



#sbatch --nodes=1 --time=96:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="resume_small_resnet_lv_3_small_imagenet.err" --gres=gpu:1 --output="resume_small_resnet_lv_3_small_imagenet.output"  --job-name="resume_small_resnet_lv_3_small_imagenet" slurm_run.sh "resnet_small" "small_imagenet" 4 3 "normal" 200 "recording_200" 1 1 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/checkpoints/resnet_small_normal_small_imagenet_seed.0_rf_level_3_recording_200_test_acc_51.46.pth"
#sbatch --nodes=1 --time=96:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="resume_small_resnet_lv_5_small_imagenet.err" --gres=gpu:1 --output="resume_small_resnet_lv_5_small_imagenet.output"  --job-name="resume_small_resnet_lv_5_small_imagenet" slurm_run.sh "resnet_small" "small_imagenet" 4 5 "normal" 200 "recording_200" 1 1 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/checkpoints/resnet_small_normal_small_imagenet_seed.0_rf_level_5_recording_200_test_acc_54.11.pth"
#sbatch --nodes=1 --time=140:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="resume_small_resnet_lv_6_small_imagenet.err" --gres=gpu:1 --output="resume_small_resnet_lv_6_small_imagenet.output"  --job-name="resume_small_resnet_lv_6_small_imagenet" slurm_run_ffcv.sh "resnet_small" "small_imagenet" 4 6 "normal" 200 "recording_200_ffcv" 1 1  "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/checkpoints/resnet_small_normal_small_imagenet_1719238673.8039715_rf_level_6_recording_200_ffcv_test_acc_45.34.pth"



############################ Iterative RF experiments #################################################

#sbatch --nodes=1 --ntasks-per-node=1 --cpus-per-task=4 --time=140:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="dynamic_RF_100_resnet50_ffcv_reverse.err" --gres=gpu:1 --output="dynamic_RF_100_resnet50_ffcv_reverse.output"  --job-name="dynamic_RF_100_resnet50_ffcv_reverse" slurm_run_ffcv.sh "resnet50" "small_imagenet" 4 7 "normal" 100 "dynamic_RF_100_version_ffcv_inverse" 1 1 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv"
#sbatch --nodes=1 --ntasks-per-node=1 --cpus-per-task=4 --time=140:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="dynamic_RF_100_vgg_ffcv_reverse.err" --gres=gpu:1 --output="dynamic_RF_100_vgg_ffcv_reverse.output"  --job-name="dynamic_RF_100_vgg_ffcv_reverse" slurm_run_ffcv.sh "vgg19" "small_imagenet" 4 7 "normal" 100 "dynamic_RF_100_version_ffcv_inverse" 1 1 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv"

#sbatch --nodes=1 --time=140:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="static_RF_100_resnet50.err" --gres=gpu:1 --output="static_RF_100_resnet50.output"  --job-name="static_RF_resnet50" slurm_run.sh "resnet50" "small_imagenet" 4 3 "normal" 100 "static_RF_100" 1 1 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv"
#sbatch --nodes=1 --time=140:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="static_RF_100_vgg.err" --gres=gpu:1 --output="static_RF_100_vgg.output"  --job-name="static_RF_vgg" slurm_run.sh "vgg19" "small_imagenet" 4 3 "normal" 100 "static_RF_100" 1 1 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv"

#####                             For one_shot pruning with specific pruning rate
#
##model="resnet_small"
##model="resnet18"
#model="resnet50"
##model="vgg19"
##dataset="small_imagenet"
#dataset="cifar10"
##directory=/nobackup/sclaam/checkpoints
##directory=/home/luisaam/Documents/PhD/checkpoints
#directory="$HOME/checkpoints"
#
###seeds=(0 1 2)
##pruning_rates=("0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9")
#pruning_rates=("0.6" "0.7" "0.8" "0.9" "0.95")
##pruning_rates=("0.9")
#
##rf_levels=(2 3)
##rf_levels=("k8")
#
##rf_levels=(1 2 3 4)
#
#rf_levels=("0.001" "0.003" "0.005")
#
##rf_levels=(1 2 3 4 'k6' 'k7' "k8")
## rf for vgg
##rf_levels=(3 4 5 "k6" "k7" "k8")
#
##rf_levels=("k6" "k7" "k8")
##rf_levels=(3 5 7)
#
#levels_max=${#rf_levels[@]}                                  # Take the length of that array
##seeds_per_level=${#list_to_use[@]}                            # Take the length of that array
#number_pruning_rates=${#pruning_rates[@]}                            # Take the length of that array
#
#for ((idxA=0; idxA<number_pruning_rates; idxA++)); do                # iterate idxA from 0 to length
#for ((idxB=0; idxB<levels_max; idxB++));do              # iterate idxB from 0 to length
#
##qsub -N "${model}_${dataset}pruning_fine_tuning_summary_level_1_${pruning_rates[$idxB]}" run.sh "${model}" "${dataset}" "2" "1" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}" "0.9" "2"
#
##qsub -l coproc_v100=1 -l h_rt=00:20:00 -N "${model}_${dataset}_pruning_summary_level_${rf_levels[$idxB]}_${pruning_rates[$idxA]}" run.sh "${model}" "${dataset}" "4" "${rf_levels[$idxB]}" "normal" "${directory}" "${pruning_rates[$idxA]}" "1"
#
#sbatch --nodes=1 --time=03:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk  --error="${model}_${rf_levels[$idxB]}_${dataset}_${pruning_rates[$idxA]}_soup.err" --output="${model}_${rf_levels[$idxB]}_${dataset}_${pruning_rates[$idxA]}_soup.out" --job-name="${model}_${rf_levels[$idxB]}_${dataset}_${pruning_rates[$idxA]}_soup"  slurm_run.sh "${pruning_rates[$idxA]}" "${model}" "${rf_levels[$idxB]}" "${dataset}"
#
##./run.sh "${model}" "${dataset}" "4" "${rf_levels[$idxB]}" "normal" "${directory}" "${pruning_rates[$idxA]}" "1"
###./run.sh "${model}" "${dataset}" "2" "1" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}" "0.5" "1"
#
#done
#done


################################################################################################
######################## Pruning one shot results ##############################################
################################################################################################
#
#run_pruning() {
#model=$1
#dataset=$2
#directory=$3
#data_folder=$4
#name=$5
#ffcv=$6
#ffcv_train=$7
#ffcv_val=$8
##pruning_rates=("0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9")
#pruning_rates=("0.5")
#
## For resnet18
#
#  if [ "${9}" -gt 0 ]
#  then
##      rf_levels=("5" "7" "8" "9")
#       rf_levels=("5")
#
#  else
#        rf_levels=("3" "4" "6" "10")
##         rf_levels=("6")
#  fi
#
#levels_max=${#rf_levels[@]}                                  # Take the length of that array
#number_pruning_rates=${#pruning_rates[@]}                            # Take the length of that array
#for ((idxA=0; idxA<number_pruning_rates; idxA++)); do                # iterate idxA from 0 to length
#for ((idxB=0; idxB<levels_max; idxB++));do              # iterate idxB from 0 to length
#
#
#if [ "${ffcv}" -gt 0 ]
#  then
#  echo "Use FFCV"
#sbatch --nodes=1 --time=03:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk  --error="${model}_${rf_levels[$idxB]}_${dataset}_${pruning_rates[$idxA]}_pruning_ffcv.err" --output="${model}_${rf_levels[$idxB]}_${dataset}_${pruning_rates[$idxA]}_pruning_ffcv.out" --job-name="${model}_${rf_levels[$idxB]}_${dataset}_${pruning_rates[$idxA]}_pruning_ffcv"   slurm_pruning_run.sh FFCV="${ffcv}" NAME="${name}" MODEL="${model}" DATASET="${dataset}"  NUMW=4  RFL="${rf_levels[$idxB]}" TYPE="normal" FOLDER="${directory}" PR="${pruning_rates[$idxA]}" EXPERIMENT=1 FFCV_TRAIN="${ffcv_train}" FFCV_VAL="${ffcv_val}" DATA_FOLDER="${data_folder}"
#
#else
# echo "Don't use FFCV"
# sbatch --nodes=1 --time=03:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk  --error="${model}_${rf_levels[$idxB]}_${dataset}_${pruning_rates[$idxA]}_pruning_no_ffcv.err" --output="${model}_${rf_levels[$idxB]}_${dataset}_${pruning_rates[$idxA]}_pruning_no_ffcv.out" --job-name="${model}_${rf_levels[$idxB]}_${dataset}_${pruning_rates[$idxA]}_pruning_no_ffcv" slurm_pruning_run.sh FFCV="${ffcv}" NAME="${name}" MODEL="${model}" DATASET="${dataset}"  NUMW=4  RFL="${rf_levels[$idxB]}" TYPE="normal" FOLDER="${directory}" PR="${pruning_rates[$idxA]}" EXPERIMENT=1 DATA_FOLDER="${data_folder}"
#  fi
#done
#done
#}
#



#run_pruning "resnet_small" "small_imagenet" "${HOME}/checkpoints_temp" "${HOME}/datasets" "recording_200_ffcv" 1 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" 0
#run_pruning "resnet_small" "small_imagenet" "${HOME}/checkpoints_arc4_2/checkpoints" "${HOME}/datasets" "recording_200" 0 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" 1

#run_pruning "resnet_small" "small_imagenet" "${HOME}/checkpoints" "${HOME}/datasets" "recording_200_ffcv" 0 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" 1
#run_pruning "resnet_small" "small_imagenet" "${HOME}/checkpoints" 0

########################################################################################################################
#################################################### HESSIAN COMPUTATION ###############################################
########################################################################################################################
# First for pyhessian

# Initialisation

#sbatch --nodes=1 --time=143:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk  --error="pyhessian_level_3_init_no_ffcv.err" --output="pyhessian_level_3_init_no_ffcv.out" --job-name="pyhessian_level_3_init_no_ffcv" slurm_hessian.sh FFCV=0 NAME="init_no_ffcv" MODEL="resnet_small" DATASET="small_imagenet" NUMW=4 RFL=3 TYPE=normal FOLDER=$HOME/hesian_results FFCV_TRAIN="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" FFCV_VAL="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" DATA_FOLDER="${HOME}/datasets" METHOD="pyhessian" BATCH_ONLY=0 N_EIGEN=300 N_BUFFER=90 SOLUTION="${HOME}/seeds_for_hessian/resnet_small_normal_small_imagenet_seed.0_rf_level_3_recording_200_no_ffcv_initial_weights.pth" EVAL_SIZE=1000
#sbatch --nodes=1 --time=143:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk  --error="pyhessian_level_4_init_no_ffcv.err" --output="pyhessian_level_4_init_no_ffcv.out" --job-name="pyhessian_level_4_init_no_ffcv" slurm_hessian.sh FFCV=0 NAME="init_no_ffcv" MODEL="resnet_small" DATASET="small_imagenet" NUMW=4 RFL=4 TYPE=normal FOLDER=$HOME/hessian_results FFCV_TRAIN="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" FFCV_VAL="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" DATA_FOLDER="${HOME}/datasets" METHOD="pyhessian" BATCH_ONLY=0 N_EIGEN=300 N_BUFFER=90 SOLUTION="${HOME}/seeds_for_hessian/resnet_small_normal_small_imagenet_seed.1_rf_level_4_recording_200_no_ffcv_initial_weights.pth" EVAL_SIZE=1000
#sbatch --nodes=1 --time=143:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk  --error="pyhessian_level_5_init.err" --output="pyhessian_level_5_init.out" --job-name="pyhessian_level_5_init" slurm_hessian.sh FFCV=0 NAME="init" MODEL="resnet_small" DATASET="small_imagenet" NUMW=4 RFL=5 TYPE=normal FOLDER=$HOME/hessian_results FFCV_TRAIN="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" FFCV_VAL="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" DATA_FOLDER="${HOME}/datasets" METHOD="pyhessian" BATCH_ONLY=0 N_EIGEN=300 N_BUFFER=90 SOLUTION="${HOME}/seeds_for_hessian/resnet_small_normal_small_imagenet_seed.5_rf_level_5_recording_200_initial_weights.pth" EVAL_SIZE=10000
#sbatch --nodes=1 --time=143:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk  --error="pyhessian_level_6_init_no_ffcv.err" --output="pyhessian_level_6_init_no_ffcv.out" --job-name="pyhessian_level_6_init_no_ffcv" slurm_hessian.sh FFCV=0 NAME="init_no_ffcv" MODEL="resnet_small" DATASET="small_imagenet" NUMW=4 RFL=6 TYPE=normal FOLDER=$HOME/hessian_results FFCV_TRAIN="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" FFCV_VAL="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" DATA_FOLDER="${HOME}/datasets" METHOD="pyhessian" BATCH_ONLY=0 N_EIGEN=300 N_BUFFER=90 SOLUTION="${HOME}/seeds_for_hessian/resnet_small_normal_small_imagenet_seed.3_rf_level_6_recording_200_ffcv_initial_weights.pth" EVAL_SIZE=1000
#sbatch --nodes=1 --time=143:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk  --error="pyhessian_level_7_init.err" --output="pyhessian_level_7_init.out" --job-name="pyhessian_level_7_init" slurm_hessian.sh FFCV=0 NAME="init" MODEL="resnet_small" DATASET="small_imagenet" NUMW=4 RFL=7 TYPE=normal FOLDER=$HOME/hessian_results FFCV_TRAIN="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" FFCV_VAL="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" DATA_FOLDER="${HOME}/datasets" METHOD="pyhessian" BATCH_ONLY=0 N_EIGEN=300 N_BUFFER=90 SOLUTION="${HOME}/seeds_for_hessian/resnet_small_normal_small_imagenet_seed.3_rf_level_7_recording_200_initial_weights.pth" EVAL_SIZE=1000
#sbatch --nodes=1 --time=143:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk  --error="pyhessian_level_8_init.err" --output="pyhessian_level_8_init.out" --job-name="pyhessian_level_8_init" slurm_hessian.sh FFCV=0 NAME="init" MODEL="resnet_small" DATASET="small_imagenet" NUMW=4 RFL=8 TYPE=normal FOLDER=$HOME/hessian_results FFCV_TRAIN="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" FFCV_VAL="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" DATA_FOLDER="${HOME}/datasets" METHOD="pyhessian" BATCH_ONLY=0 N_EIGEN=300 N_BUFFER=90 SOLUTION="${HOME}/seeds_for_hessian/resnet_small_normal_small_imagenet_seed.6_rf_level_8_recording_200_initial_weights.pth" EVAL_SIZE=1000
#sbatch --nodes=1 --time=143:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk  --error="pyhessian_level_9_init.err" --output="pyhessian_level_9_init.out" --job-name="pyhessian_level_9_init" slurm_hessian.sh FFCV=0 NAME="init" MODEL="resnet_small" DATASET="small_imagenet" NUMW=4 RFL=9 TYPE=normal FOLDER=$HOME/hessian_results FFCV_TRAIN="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" FFCV_VAL="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" DATA_FOLDER="${HOME}/datasets" METHOD="pyhessian" BATCH_ONLY=0 N_EIGEN=300 N_BUFFER=90 SOLUTION="${HOME}/seeds_for_hessian/resnet_small_normal_small_imagenet_seed.7_rf_level_9_recording_200_initial_weights.pth" EVAL_SIZE=1000
#sbatch --nodes=1 --time=143:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk  --error="pyhessian_level_10_init_no_ffcv.err" --output="pyhessian_level_10_init_no_ffcv.out" --job-name="pyhessian_level_10_init_no_ffcv" slurm_hessian.sh FFCV=0 NAME="init_no_ffcv" MODEL="resnet_small" DATASET="small_imagenet" NUMW=4 RFL=10 TYPE=normal FOLDER=$HOME/hessian_results FFCV_TRAIN="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" FFCV_VAL="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" DATA_FOLDER="${HOME}/datasets" METHOD="pyhessian" BATCH_ONLY=0 N_EIGEN=300 N_BUFFER=90 SOLUTION="${HOME}/seeds_for_hessian/resnet_small_normal_small_imagenet_seed.4_rf_level_10_recording_200_ffcv_initial_weights.pth" EVAL_SIZE=1000

# Trained

#sbatch --nodes=1 --time=143:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk  --error="pyhessian_level_3_trained.err" --output="pyhessian_level_3_trained.out" --job-name="pyhessian_level_3_trained" slurm_hessian.sh FFCV=0 NAME="trained_no_ffcv" MODEL="resnet_small" DATASET="small_imagenet" NUMW=4 RFL=3 TYPE=normal FOLDER=$HOME/hessian_results FFCV_TRAIN="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" FFCV_VAL="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" DATA_FOLDER="${HOME}/datasets" METHOD="pyhessian" BATCH_ONLY=0 N_EIGEN=300 N_BUFFER=90 SOLUTION="${HOME}/seeds_for_hessian/resnet_small_normal_small_imagenet_seed.0_rf_level_3_recording_200_no_ffcv_test_acc_58.64.pth" EVAL_SIZE=1000
#sbatch --nodes=1 --time=143:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk  --error="pyhessian_level_4_trained.err" --output="pyhessian_level_4_trained.out" --job-name="pyhessian_level_4_trained" slurm_hessian.sh FFCV=0 NAME="trained_no_ffcv" MODEL="resnet_small" DATASET="small_imagenet" NUMW=4 RFL=4 TYPE=normal FOLDER=$HOME/hessian_results FFCV_TRAIN="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" FFCV_VAL="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" DATA_FOLDER="${HOME}/datasets" METHOD="pyhessian" BATCH_ONLY=0 N_EIGEN=300 N_BUFFER=90 SOLUTION="${HOME}/seeds_for_hessian/resnet_small_normal_small_imagenet_seed.1_rf_level_4_recording_200_no_ffcv_test_acc_60.47.pth" EVAL_SIZE=1000
#sbatch --nodes=1 --time=143:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk  --error="pyhessian_level_5_trained.err" --output="pyhessian_level_5_trained.out" --job-name="pyhessian_level_5_trained" slurm_hessian.sh FFCV=0 NAME="trained" MODEL="resnet_small" DATASET="small_imagenet" NUMW=4 RFL=5 TYPE=normal FOLDER=$HOME/hessian_results FFCV_TRAIN="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" FFCV_VAL="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" DATA_FOLDER="${HOME}/datasets" METHOD="pyhessian" BATCH_ONLY=0 N_EIGEN=300 N_BUFFER=90 SOLUTION="${HOME}/seeds_for_hessian/resnet_small_normal_small_imagenet_seed.5_rf_level_5_recording_200_test_acc_61.97.pth" EVAL_SIZE=1000
#sbatch --nodes=1 --time=143:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk  --error="pyhessian_level_6_trained.err" --output="pyhessian_level_6_trained.out" --job-name="pyhessian_level_6_trained" slurm_hessian.sh FFCV=0 NAME="trained_no_ffcv" MODEL="resnet_small" DATASET="small_imagenet" NUMW=4 RFL=6 TYPE=normal FOLDER=$HOME/hessian_results FFCV_TRAIN="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" FFCV_VAL="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" DATA_FOLDER="${HOME}/datasets" METHOD="pyhessian" BATCH_ONLY=0 N_EIGEN=300 N_BUFFER=90 SOLUTION="${HOME}/seeds_for_hessian/resnet_small_normal_small_imagenet_seed.3_rf_level_6_recording_200_ffcv_test_acc_57.83.pth" EVAL_SIZE=1000
#sbatch --nodes=1 --time=143:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk  --error="pyhessian_level_7_trained.err" --output="pyhessian_level_7_trained.out" --job-name="pyhessian_level_7_trained" slurm_hessian.sh FFCV=0 NAME="trained" MODEL="resnet_small" DATASET="small_imagenet" NUMW=4 RFL=7 TYPE=normal FOLDER=$HOME/hessian_results FFCV_TRAIN="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" FFCV_VAL="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" DATA_FOLDER="${HOME}/datasets" METHOD="pyhessian" BATCH_ONLY=0 N_EIGEN=300 N_BUFFER=90 SOLUTION="${HOME}/seeds_for_hessian/resnet_small_normal_small_imagenet_seed.3_rf_level_7_recording_200_test_acc_51.66.pth" EVAL_SIZE=1000
#sbatch --nodes=1 --time=143:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk  --error="pyhessian_level_8_trained.err" --output="pyhessian_level_8_trained.out" --job-name="pyhessian_level_8_trained" slurm_hessian.sh FFCV=0 NAME="trained" MODEL="resnet_small" DATASET="small_imagenet" NUMW=4 RFL=8 TYPE=normal FOLDER=$HOME/hessian_results FFCV_TRAIN="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" FFCV_VAL="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" DATA_FOLDER="${HOME}/datasets" METHOD="pyhessian" BATCH_ONLY=0 N_EIGEN=300 N_BUFFER=90 SOLUTION="${HOME}/seeds_for_hessian/resnet_small_normal_small_imagenet_seed.6_rf_level_8_recording_200_test_acc_49.49.pth" EVAL_SIZE=1000
#sbatch --nodes=1 --time=143:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk  --error="pyhessian_level_9_trained.err" --output="pyhessian_level_9_trained.out" --job-name="pyhessian_level_9_trained" slurm_hessian.sh FFCV=0 NAME="trained" MODEL="resnet_small" DATASET="small_imagenet" NUMW=4 RFL=9 TYPE=normal FOLDER=$HOME/hessian_results FFCV_TRAIN="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" FFCV_VAL="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" DATA_FOLDER="${HOME}/datasets" METHOD="pyhessian" BATCH_ONLY=0 N_EIGEN=300 N_BUFFER=90 SOLUTION="${HOME}/seeds_for_hessian/resnet_small_normal_small_imagenet_seed.7_rf_level_9_recording_200_test_acc_45.04.pth" EVAL_SIZE=1000
#sbatch --nodes=1 --time=143:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk  --error="pyhessian_level_10_trained.err" --output="pyhessian_level_10_trained.out" --job-name="pyhessian_level_10_trained" slurm_hessian.sh FFCV=0 NAME="trained_no_ffcv" MODEL="resnet_small" DATASET="small_imagenet" NUMW=4 RFL=10 TYPE=normal FOLDER=$HOME/hessian_results FFCV_TRAIN="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" FFCV_VAL="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" DATA_FOLDER="${HOME}/datasets" METHOD="pyhessian" BATCH_ONLY=0 N_EIGEN=300 N_BUFFER=90 SOLUTION="${HOME}/seeds_for_hessian/resnet_small_normal_small_imagenet_seed.4_rf_level_10_recording_200_no_ffcv_test_acc_39.98.pth" EVAL_SIZE=1000

# Now for torchessian

# Initialisation

#sbatch --nodes=1 --time=143:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk  --error="torchessian_level_3_init.err" --output="torchessian_level_3_init.out" --job-name="torchessian_level_3_init" slurm_hessian.sh FFCV=0 NAME="init_ffcv" MODEL="resnet_small" DATASET="small_imagenet" NUMW=4 RFL=3 TYPE=normal FOLDER=$HOME/hessian_results FFCV_TRAIN="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" FFCV_VAL="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" DATA_FOLDER="${HOME}/datasets" METHOD="torchessian" BATCH_ONLY=0 N_EIGEN=300 N_BUFFER=2 SOLUTION="${HOME}/seeds_for_hessian/resnet_small_normal_small_imagenet_seed.0_rf_level_3_recording_200_no_ffcv_initial_weights.pth" EVAL_SIZE=5000
#sbatch --nodes=1 --time=143:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk  --error="torchessian_level_4_init.err" --output="torchessian_level_4_init.out" --job-name="torchessian_level_4_init" slurm_hessian.sh FFCV=0 NAME="init_ffcv" MODEL="resnet_small" DATASET="small_imagenet" NUMW=4 RFL=4 TYPE=normal FOLDER=$HOME/hessian_results FFCV_TRAIN="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" FFCV_VAL="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" DATA_FOLDER="${HOME}/datasets" METHOD="torchessian" BATCH_ONLY=0 N_EIGEN=300 N_BUFFER=2 SOLUTION="${HOME}/seeds_for_hessian/resnet_small_normal_small_imagenet_seed.1_rf_level_4_recording_200_no_ffcv_initial_weights.pth" EVAL_SIZE=5000
#sbatch --nodes=1 --time=143:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk  --error="torchessian_level_5_init.err" --output="torchessian_level_5_init.out" --job-name="torchessian_level_5_init" slurm_hessian.sh FFCV=0 NAME="init_ffcv" MODEL="resnet_small" DATASET="small_imagenet" NUMW=4 RFL=5 TYPE=normal FOLDER=$HOME/hessian_results FFCV_TRAIN="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" FFCV_VAL="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" DATA_FOLDER="${HOME}/datasets" METHOD="torchessian" BATCH_ONLY=0 N_EIGEN=300 N_BUFFER=2 SOLUTION="${HOME}/seeds_for_hessian/resnet_small_normal_small_imagenet_seed.5_rf_level_5_recording_200_initial_weights.pth" EVAL_SIZE=5000
#sbatch --nodes=1 --time=143:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk  --error="torchessian_level_6_init.err" --output="torchessian_level_6_init.out" --job-name="torchessian_level_6_init" slurm_hessian.sh FFCV=0 NAME="init" MODEL="resnet_small" DATASET="small_imagenet" NUMW=4 RFL=6 TYPE=normal FOLDER=$HOME/hessian_results FFCV_TRAIN="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" FFCV_VAL="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" DATA_FOLDER="${HOME}/datasets" METHOD="torchessian" BATCH_ONLY=0 N_EIGEN=300 N_BUFFER=2 SOLUTION="${HOME}/seeds_for_hessian/resnet_small_normal_small_imagenet_seed.3_rf_level_6_recording_200_ffcv_initial_weights.pth" EVAL_SIZE=5000
#sbatch --nodes=1 --time=143:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk  --error="torchessian_level_7_init.err" --output="torchessian_level_7_init.out" --job-name="torchessian_level_7_init" slurm_hessian.sh FFCV=0 NAME="init" MODEL="resnet_small" DATASET="small_imagenet" NUMW=4 RFL=7 TYPE=normal FOLDER=$HOME/hessian_results FFCV_TRAIN="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" FFCV_VAL="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" DATA_FOLDER="${HOME}/datasets" METHOD="torchessian" BATCH_ONLY=0 N_EIGEN=300 N_BUFFER=2 SOLUTION="${HOME}/seeds_for_hessian/resnet_small_normal_small_imagenet_seed.3_rf_level_7_recording_200_initial_weights.pth" EVAL_SIZE=5000
#sbatch --nodes=1 --time=143:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk  --error="torchessian_level_8_init.err" --output="torchessian_level_8_init.out" --job-name="torchessian_level_8_init" slurm_hessian.sh FFCV=0 NAME="init" MODEL="resnet_small" DATASET="small_imagenet" NUMW=4 RFL=8 TYPE=normal FOLDER=$HOME/hessian_results FFCV_TRAIN="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" FFCV_VAL="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" DATA_FOLDER="${HOME}/datasets" METHOD="torchessian" BATCH_ONLY=0 N_EIGEN=300 N_BUFFER=2 SOLUTION="${HOME}/seeds_for_hessian/resnet_small_normal_small_imagenet_seed.6_rf_level_8_recording_200_initial_weights.pth" EVAL_SIZE=5000
#sbatch --nodes=1 --time=143:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk  --error="torchessian_level_9_init.err" --output="torchessian_level_9_init.out" --job-name="torchessian_level_9_init" slurm_hessian.sh FFCV=0 NAME="init" MODEL="resnet_small" DATASET="small_imagenet" NUMW=4 RFL=9 TYPE=normal FOLDER=$HOME/hessian_results FFCV_TRAIN="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" FFCV_VAL="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" DATA_FOLDER="${HOME}/datasets" METHOD="torchessian" BATCH_ONLY=0 N_EIGEN=300 N_BUFFER=2 SOLUTION="${HOME}/seeds_for_hessian/resnet_small_normal_small_imagenet_seed.7_rf_level_9_recording_200_initial_weights.pth" EVAL_SIZE=5000
#sbatch --nodes=1 --time=143:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk  --error="torchessian_level_10_init.err" --output="torchessian_level_10_init.out" --job-name="torchessian_level_10_init" slurm_hessian.sh FFCV=0 NAME="init_ffcv" MODEL="resnet_small" DATASET="small_imagenet" NUMW=4 RFL=10 TYPE=normal FOLDER=$HOME/hessian_results FFCV_TRAIN="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" FFCV_VAL="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" DATA_FOLDER="${HOME}/datasets" METHOD="torchessian" BATCH_ONLY=0 N_EIGEN=300 N_BUFFER=2 SOLUTION="${HOME}/seeds_for_hessian/resnet_small_normal_small_imagenet_seed.4_rf_level_10_recording_200_ffcv_initial_weights.pth" EVAL_SIZE=5000

## Trained

#sbatch --nodes=1 --time=143:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk  --error="torchessian_level_3_trained.err" --output="torchessian_level_3_trained.out" --job-name="torchessian_level_3_trained" slurm_hessian.sh FFCV=0 NAME="trained_ffcv" MODEL="resnet_small" DATASET="small_imagenet" NUMW=4 RFL=3 TYPE=normal FOLDER=$HOME/hessian_results FFCV_TRAIN="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" FFCV_VAL="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" DATA_FOLDER="${HOME}/datasets" METHOD="torchessian" BATCH_ONLY=0 N_EIGEN=300 N_BUFFER=2 SOLUTION="${HOME}/seeds_for_hessian/resnet_small_normal_small_imagenet_seed.0_rf_level_3_recording_200_no_ffcv_test_acc_58.64.pth" EVAL_SIZE=5000
#sbatch --nodes=1 --time=143:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk  --error="torchessian_level_4_trained.err" --output="torchessian_level_4_trained.out" --job-name="torchessian_level_4_trained" slurm_hessian.sh FFCV=0 NAME="trained_ffcv" MODEL="resnet_small" DATASET="small_imagenet" NUMW=4 RFL=4 TYPE=normal FOLDER=$HOME/hessian_results FFCV_TRAIN="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" FFCV_VAL="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" DATA_FOLDER="${HOME}/datasets" METHOD="torchessian" BATCH_ONLY=0 N_EIGEN=300 N_BUFFER=2 SOLUTION="${HOME}/seeds_for_hessian/resnet_small_normal_small_imagenet_seed.1_rf_level_4_recording_200_no_ffcv_test_acc_60.47.pth" EVAL_SIZE=5000
#sbatch --nodes=1 --time=143:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk  --error="torchessian_level_5_trained.err" --output="torchessian_level_5_trained.out" --job-name="torchessian_level_5_trained" slurm_hessian.sh FFCV=0 NAME="trained" MODEL="resnet_small" DATASET="small_imagenet" NUMW=4 RFL=5 TYPE=normal FOLDER=$HOME/hessian_results FFCV_TRAIN="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" FFCV_VAL="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" DATA_FOLDER="${HOME}/datasets" METHOD="torchessian" BATCH_ONLY=0 N_EIGEN=300 N_BUFFER=2 SOLUTION="${HOME}/seeds_for_hessian/resnet_small_normal_small_imagenet_seed.5_rf_level_5_recording_200_test_acc_61.97.pth" EVAL_SIZE=5000
#sbatch --nodes=1 --time=143:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk  --error="torchessian_level_6_trained.err" --output="torchessian_level_6_trained.out" --job-name="torchessian_level_6_trained" slurm_hessian.sh FFCV=0 NAME="trained_ffcv" MODEL="resnet_small" DATASET="small_imagenet" NUMW=4 RFL=6 TYPE=normal FOLDER=$HOME/hessian_results FFCV_TRAIN="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" FFCV_VAL="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" DATA_FOLDER="${HOME}/datasets" METHOD="torchessian" BATCH_ONLY=0 N_EIGEN=300 N_BUFFER=2 SOLUTION="${HOME}/seeds_for_hessian/resnet_small_normal_small_imagenet_seed.3_rf_level_6_recording_200_ffcv_test_acc_57.83.pth" EVAL_SIZE=5000
#sbatch --nodes=1 --time=143:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk  --error="torchessian_level_7_trained.err" --output="torchessian_level_7_trained.out" --job-name="torchessian_level_7_trained" slurm_hessian.sh FFCV=0 NAME="trained" MODEL="resnet_small" DATASET="small_imagenet" NUMW=4 RFL=7 TYPE=normal FOLDER=$HOME/hessian_results FFCV_TRAIN="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" FFCV_VAL="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" DATA_FOLDER="${HOME}/datasets" METHOD="torchessian" BATCH_ONLY=0 N_EIGEN=300 N_BUFFER=2 SOLUTION="${HOME}/seeds_for_hessian/resnet_small_normal_small_imagenet_seed.3_rf_level_7_recording_200_test_acc_51.66.pth" EVAL_SIZE=5000
#sbatch --nodes=1 --time=143:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk  --error="torchessian_level_8_trained.err" --output="torchessian_level_8_trained.out" --job-name="torchessian_level_8_trained" slurm_hessian.sh FFCV=0 NAME="trained" MODEL="resnet_small" DATASET="small_imagenet" NUMW=4 RFL=8 TYPE=normal FOLDER=$HOME/hessian_results FFCV_TRAIN="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" FFCV_VAL="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" DATA_FOLDER="${HOME}/datasets" METHOD="torchessian" BATCH_ONLY=0 N_EIGEN=300 N_BUFFER=2 SOLUTION="${HOME}/seeds_for_hessian/resnet_small_normal_small_imagenet_seed.6_rf_level_8_recording_200_test_acc_49.49.pth" EVAL_SIZE=5000
#sbatch --nodes=1 --time=143:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk  --error="torchessian_level_9_trained.err" --output="torchessian_level_9_trained.out" --job-name="torchessian_level_9_trained" slurm_hessian.sh FFCV=0 NAME="trained" MODEL="resnet_small" DATASET="small_imagenet" NUMW=4 RFL=9 TYPE=normal FOLDER=$HOME/hessian_results FFCV_TRAIN="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" FFCV_VAL="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" DATA_FOLDER="${HOME}/datasets" METHOD="torchessian" BATCH_ONLY=0 N_EIGEN=300 N_BUFFER=2 SOLUTION="${HOME}/seeds_for_hessian/resnet_small_normal_small_imagenet_seed.7_rf_level_9_recording_200_test_acc_45.04.pth" EVAL_SIZE=5000
#sbatch --nodes=1 --time=143:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk  --error="torchessian_level_10_trained.err" --output="torchessian_level_10_trained.out" --job-name="torchessian_level_10_trained" slurm_hessian.sh FFCV=0 NAME="trained_ffcv" MODEL="resnet_small" DATASET="small_imagenet" NUMW=4 RFL=10 TYPE=normal FOLDER=$HOME/hessian_results FFCV_TRAIN="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" FFCV_VAL="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" DATA_FOLDER="${HOME}/datasets" METHOD="torchessian" BATCH_ONLY=0 N_EIGEN=300 N_BUFFER=2 SOLUTION="${HOME}/seeds_for_hessian/resnet_small_normal_small_imagenet_seed.4_rf_level_10_recording_200_no_ffcv_test_acc_39.98.pth" EVAL_SIZE=5000



#####################################################################################
################################ Stochastic pruning soup idea #######################
#####################################################################################


run_soup_stochastic() {
model=$1
dataset=$2

echo "model ${model} and dataset ${dataset}"

#pruning_rates=("0.5" "0.6" "0.7" "0.8" "0.9" "0.95")

pruning_rates=("0.5")

# For resnet18

if [ "${3}" -eq 1 ]; then

#    rf_levels=("0.001" "0.003" "0.005")

    rf_levels=("0.005")
else

    rf_levels=("0.007" "0.008" "0.01")
fi

#pruners=("global" "lamp")

pruners=("lamp")

pruners_max=${#pruners[@]}                                  # Take the length of that array
levels_max=${#rf_levels[@]}                                  # Take the length of that array
number_pruning_rates=${#pruning_rates[@]}                            # Take the length of that array
#echo "About to enter the for loops"
for ((idxA=0; idxA<number_pruning_rates; idxA++)); do                # iterate idxA from 0 to length
#echo "Entered the pruning rate loop"
for ((idxB=0; idxB<levels_max; idxB++));do              # iterate idxB from 0 to length
#echo "Entered the level for loop"
for ((idxC=0; idxC<pruners_max; idxC++));do              # iterate idxB from 0 to length
#  echo "Entered the pruners loop"
#qsub -l coproc_v100=1 -l h_rt=02:00:00 -N "${model}_${dataset}_soup_idea_${rf_levels[$idxB]}_${pruning_rates[$idxA]}" run.sh "${pruning_rates[$idxA]}" "${model}" "${rf_levels[$idxB]}" "${dataset}"
#echo "Entered in the loop!"
  echo "${rf_levels[$idxB]} ${pruning_rates[$idxA]} ${pruners[$idxC]}"
  sbatch --nodes=1 --time=01:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="${model}_${dataset}_soup_idea_${rf_levels[$idxB]}_${pruning_rates[$idxA]}_${pruners[$idxC]}.err" --output="${model}_${dataset}_soup_idea_${rf_levels[$idxB]}_${pruning_rates[$idxA]}_${pruners[$idxC]}.out" --job-name="${model}_${dataset}_soup_idea_${rf_levels[$idxB]}_${pruning_rates[$idxA]}_${pruners[$idxC]}" slurm_run_ffcv.sh "${pruning_rates[$idxA]}" "${model}" "${rf_levels[$idxB]}" "${dataset}" "${pruners[$idxC]}"

done
done
done

}
#
#run_soup_stochastic resnet18 cifar10 0
##run_soup_stochastic resnet18 cifar100 0
#run_soup_stochastic resnet50 cifar10 0
#run_soup_stochastic resnet50 cifar100 0
#run_soup_stochastic vgg19 cifar10 0
#run_soup_stochastic vgg19 cifar100 0

run_soup_stochastic resnet18 cifar10 1
#run_soup_stochastic resnet18 cifar100 1


#run_soup_stochastic resnet50 cifar10 1
#run_soup_stochastic resnet50 cifar100 1
#run_soup_stochastic vgg19 cifar10 1
#run_soup_stochastic vgg19 cifar100 1
#run_soup_stochastic vgg19 cifar100 1



