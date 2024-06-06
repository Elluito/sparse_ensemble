#!/bin/bash
#
#sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="kfac_optim_rf_1_cifar10_rs.err" --output="kfac_optim_rf_1_cifar10_rs.output"  --job-name="Kfac_optim_rf_1_cifar10_rs" slurm_2nd_order.sh "cifar10" "resnet50" "1" "normal" "kfac_optim_hyper_100" "1"
#sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="kfac_optim_rf_2_cifar10_rs.err" --output="kfac_optim_rf_2_cifar10_rs.output" --job-name="Kfac_optim_rf_2_cifar10_rs" slurm_2nd_order.sh "cifar10" "resnet50" "2" "normal" "kfac_optim_hyper_100" "1"
#sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="kfac_optim_rf_3_cifar10_rs.err" --output="kfac_optim_rf_3_cifar10_rs.output"  --job-name="Kfac_optim_rf_3_cifar10_rs" slurm_2nd_order.sh "cifar10" "resnet50" "3" "normal" "kfac_optim_hyper_100" "1"
#sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="kfac_optim_rf_4_cifar10_rs.err" --output="kfac_optim_rf_4_cifar10_rs.output" --job-name="Kfac_optim_rf_4_cifar10_rs" slurm_2nd_order.sh "cifar10" "resnet50" "4" "normal" "kfac_optim_hyper_100" "1"
#sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="kfac_optim_rf_k6_cifar10_rs.err" --output="kfac_optim_rf_k6_cifar10_rs.output" --job-name="Kfac_optim_rf_k6_cifar10_rs" slurm_2nd_order.sh "cifar10" "resnet50" "k6" "normal" "kfac_optim_hyper_100" "1"
#sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="kfac_optim_rf_k7_cifar10_rs.err" --output="kfac_optim_rf_k7_cifar10_rs.output" --job-name="Kfac_optim_rf_k7_cifar10_rs" slurm_2nd_order.sh "cifar10" "resnet50" "k7" "normal" "kfac_optim_hyper_100" "1"
#sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="kfac_optim_rf_k8_cifar10_rs.err" --output="kfac_optim_rf_k8_cifar10_rs.output" --job-name="Kfac_optim_rf_k8_cifar10_rs" slurm_2nd_order.sh "cifar10" "resnet50" "k8" "normal" "kfac_optim_hyper_100" "1"
#sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="kfac_optim_rf_k9_cifar10_rs.err" --output="kfac_optim_rf_k9_cifar10_rs.output" --job-name="Kfac_optim_rf_k9_cifar10_rs" slurm_2nd_order.sh "cifar10" "resnet50" "k9" "normal" "kfac_optim_hyper_100" "1"
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
#                                                   FFCV variants
#sbatch --nodes=1 --time=96:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="small_resnet_lv_8_small_imagenet_ffcv.err" --gres=gpu:1 --output="small_resnet_lv_8_small_imagenet_ffcv.output"  --job-name="small_resnet_lv_8_small_imagenet_ffcv" slurm_run_ffcv.sh "resnet_small" "small_imagenet" 1 8 "normal" 200 "recording_200_ffcv" 1 1 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv"
#sbatch --nodes=1 --time=96:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="small_resnet_lv_9_small_imagenet_ffcv.err" --gres=gpu:1 --output="small_resnet_lv_9_small_imagenet_ffcv.output"  --job-name="small_resnet_lv_9_small_imagenet_ffcv" slurm_run_ffcv.sh "resnet_small" "small_imagenet" 1 9 "normal" 200 "recording_200_ffcv" 1 1 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv"
#sbatch --nodes=1 --time=96:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="small_resnet_lv_10_small_imagenet_ffcv.err" --gres=gpu:1 --output="small_resnet_lv_10_small_imagenet_ffcv.output"  --job-name="small_resnet_lv_10_small_imagenet_ffcv" slurm_run_ffcv.sh "resnet_small" "small_imagenet" 1 10 "normal" 200 "recording_200_ffcv" 1 1 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv"


############################## Resume small resnet #######################################################
#qsub -l h_rt=48:00:00 -l coproc_v100=1 -N "resume_training_Level_3_small_resnet_small_imagenet" resume_run.sh "resnet_small" "small_imagenet" 2 3 "normal" 200 "recording_200" 1 1 "/nobackup/sclaam/checkpoints/resnet_small_normal_small_imagenet_seed.0_rf_level_3_recording_200_test_acc_42.89.pth"
#sbatch --nodes=1 --time=96:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="resume_small_resnet_lv_3_small_imagenet.err" --gres=gpu:1 --output="resume_small_resnet_lv_3_small_imagenet.output"  --job-name="resume_small_resnet_lv_3_small_imagenet" slurm_run.sh "resnet_small" "small_imagenet" 1 3 "normal" 200 "recording_200" 1 1 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/checkpoints/resnet_small_normal_small_imagenet_seed.0_rf_level_3_recording_200_test_acc_51.46.pth"
#sbatch --nodes=1 --time=96:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="resume_small_resnet_lv_5_small_imagenet.err" --gres=gpu:1 --output="resume_small_resnet_lv_5_small_imagenet.output"  --job-name="resume_small_resnet_lv_5_small_imagenet" slurm_run.sh "resnet_small" "small_imagenet" 1 5 "normal" 200 "recording_200" 1 1 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/checkpoints/resnet_small_normal_small_imagenet_seed.0_rf_level_5_recording_200_test_acc_54.11.pth"
#sbatch --nodes=1 --time=96:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="resume_small_resnet_lv_7_small_imagenet.err" --gres=gpu:1 --output="resume_small_resnet_lv_7_small_imagenet.output"  --job-name="resume_small_resnet_lv_7_small_imagenet" slurm_run.sh "resnet_small" "small_imagenet" 1 7 "normal" 200 "recording_200" 1 1 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/checkpoints/resnet_small_normal_small_imagenet_seed.0_rf_level_7_recording_200_test_acc_43.93.pth"

#####                             For one_shot pruning with specific pruning rate
#model="resnet_small"
model="resnet18"
dataset="small_imagenet"
#directory=/nobackup/sclaam/checkpoints
#directory=/home/luisaam/Documents/PhD/checkpoints
directory=$HOME/checkpoints_arc4

##seeds=(0 1 2)
pruning_rates=("0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9")

rf_levels=(2 3)
#rf_levels=(2 3 4 'k6' 'k7' "k8")
# rf for vgg
#rf_levels=(3 4 5 "k6" "k7 "k8")

levels_max=${#rf_levels[@]}                                  # Take the length of that array
#seeds_per_level=${#list_to_use[@]}                            # Take the length of that array
number_pruning_rates=${#pruning_rates[@]}                            # Take the length of that array

for ((idxA=0; idxA<number_pruning_rates; idxA++)); do                # iterate idxA from 0 to length
for ((idxB=0; idxB<levels_max; idxB++));do              # iterate idxB from 0 to length

#qsub -N "${model}_${dataset}pruning_fine_tuning_summary_level_1_${pruning_rates[$idxB]}" run.sh "${model}" "${dataset}" "2" "1" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}" "0.9" "2"

#qsub -l coproc_v100=1 -l h_rt=00:20:00 -N "${model}_${dataset}_pruning_summary_level_${rf_levels[$idxB]}_${pruning_rates[$idxA]}" run.sh "${model}" "${dataset}" "4" "${rf_levels[$idxB]}" "normal" "${directory}" "${pruning_rates[$idxA]}" "1"
sbatch --nodes=1 --time=02:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk  --error="${model}_${rf_levels[$idxB]}_${dataset}_${pruning_rates[$idxA]}.err" --output="${model}_${rf_levels[$idxB]}_${dataset}_${pruning_rates[$idxA]}.out" --job-name="${model}_${rf_levels[$idxB]}_${dataset}_${pruning_rates[$idxA]}_one_shot_pruning"  slurm_run.sh "${model}" "${dataset}" "4" "${rf_levels[$idxB]}" "normal" "${directory}" "${pruning_rates[$idxA]}" "1"

#./run.sh "${model}" "${dataset}" "4" "${rf_levels[$idxB]}" "normal" "${directory}" "${pruning_rates[$idxA]}" "1"
##./run.sh "${model}" "${dataset}" "2" "1" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}" "0.5" "1"

done
done
