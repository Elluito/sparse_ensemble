#!/bin/bash

CHECK_1="$HOME/checkpoints_temp"
CHECK_2="$HOME/checkpoints_arc4_2/checkpoints"
# create features for later similarity comparison
# Level 3  Models
sbatch --nodes=1 --time=12:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="saving_features_lvl3_s_1.err" --output="saving_features_lvl3_s_1.out"  --job-name="saving_features_lvl3_s_1" slurm_similarity_run.sh "resnet_small" "small_imagenet" "${CHECK_1}/resnet_small_normal_small_imagenet_seed.2_rf_level_3_recording_200_no_ffcv_test_acc_58.11.pth" "_seed_2" 3 "normal" 1000 "${HOME}/features" "${HOME}/datasets" 1
sbatch --nodes=1 --time=12:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="saving_features_lvl3_s_2.err" --output="saving_features_lvl3_s_2.out"  --job-name="saving_features_lvl3_s_1" slurm_similarity_run.sh "resnet_small" "small_imagenet" "${CHECK_1}/resnet_small_normal_small_imagenet_seed.1_rf_level_3_recording_200_no_ffcv_test_acc_58.71.pth" "_seed_1" 3 "normal" 1000 "${HOME}/features" "${HOME}/datasets" 1



# Level 4  Models
sbatch --nodes=1 --time=12:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="saving_features_lvl4_s_1.err" --output="saving_features_lvl4_s_1.out"  --job-name="saving_features_lvl4_s_1" slurm_similarity_run.sh "resnet_small" "small_imagenet" "${CHECK_1}/resnet_small_normal_small_imagenet_seed.2_rf_level_4_recording_200_no_ffcv_test_acc_60.31.pth" "_seed_1" 4 "normal" 1000 "${HOME}/features" "${HOME}/datasets" 1
sbatch --nodes=1 --time=12:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="saving_features_lvl4_s_2.err" --output="saving_features_lvl4_s_2.out"  --job-name="saving_features_lvl4_s_2" slurm_similarity_run.sh "resnet_small" "small_imagenet" "${CHECK_1}/resnet_small_normal_small_imagenet_seed.1_rf_level_4_recording_200_ffcv_test_acc_60.47.pth" "_seed_2" 4 "normal" 1000 "${HOME}/features" "${HOME}/datasets" 1

# Level 5  Models

sbatch --nodes=1 --time=12:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="saving_features_lvl5_s_1.err" --output="saving_features_lvl5_s_1.out"  --job-name="saving_features_lvl5_s_1" slurm_similarity_run.sh "resnet_small" "small_imagenet" "${CHECK_2}/resnet_small_normal_small_imagenet_seed.5_rf_level_5_recording_200_test_acc_61.97.pth" "_seed_5" 5 "normal" 1000 "${HOME}/features" "${HOME}/datasets" 0
sbatch --nodes=1 --time=12:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="saving_features_lvl5_s_2.err" --output="saving_features_lvl5_s_2.out"  --job-name="saving_features_lvl5_s_2" slurm_similarity_run.sh "resnet_small" "small_imagenet" "${CHECK_2}/resnet_small_normal_small_imagenet_seed.8_rf_level_5_recording_200_test_acc_62.13.pth" "_seed_8" 5 "normal" 1000 "${HOME}/features" "${HOME}/datasets" 0

# Level 6  Models

sbatch --nodes=1 --time=12:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="saving_features_lvl6_s_1.err" --output="saving_features_lvl6_s_1.out"  --job-name="saving_features_lvl6_s_1" slurm_similarity_run.sh "resnet_small" "small_imagenet" "${CHECK_1}/resnet_small_normal_small_imagenet_seed.3_rf_level_6_recording_200_ffcv_test_acc_57.83.pth" "_seed_3" 6 "normal" 1000 "${HOME}/features" "${HOME}/datasets" 1
sbatch --nodes=1 --time=12:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="saving_features_lvl6_s_2.err" --output="saving_features_lvl6_s_2.out"  --job-name="saving_features_lvl6_s_2" slurm_similarity_run.sh "resnet_small" "small_imagenet" "${CHECK_1}/resnet_small_normal_small_imagenet_seed.2_rf_level_6_recording_200_ffcv_test_acc_58.33.pth" "_seed_2" 6 "normal" 1000 "${HOME}/features" "${HOME}/datasets" 1

# Level 7  Models

sbatch --nodes=1 --time=12:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="saving_features_lvl7_s_1.err" --output="saving_features_lvl7_s_1.out"  --job-name="saving_features_lvl7_s_1" slurm_similarity_run.sh "resnet_small" "small_imagenet" "${CHECK_2}/resnet_small_normal_small_imagenet_seed.9_rf_level_7_recording_200_test_acc_51.51.pth" "_seed_9" 7 "normal" 1000 "${HOME}/features" "${HOME}/datasets" 0
sbatch --nodes=1 --time=12:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="saving_features_lvl7_s_2.err" --output="saving_features_lvl7_s_2.out"  --job-name="saving_features_lvl7_s_2" slurm_similarity_run.sh "resnet_small" "small_imagenet" "${CHECK_2}/resnet_small_normal_small_imagenet_seed.3_rf_level_7_recording_200_test_acc_51.66.pth" "_seed_3" 7 "normal" 1000 "${HOME}/features" "${HOME}/datasets" 0

# Level 8  Models

sbatch --nodes=1 --time=12:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="saving_features_lvl8_s_1.err" --output="saving_features_lvl8_s_1.out"  --job-name="saving_features_lvl8_s_1" slurm_similarity_run.sh "resnet_small" "small_imagenet" "${CHECK_2}/resnet_small_normal_small_imagenet_seed.4_rf_level_8_recording_200_test_acc_49.1.pth" "_seed_4" 8 "normal" 1000 "${HOME}/features" "${HOME}/datasets" 0
sbatch --nodes=1 --time=12:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="saving_features_lvl8_s_2.err" --output="saving_features_lvl8_s_2.out"  --job-name="saving_features_lvl8_s_2" slurm_similarity_run.sh "resnet_small" "small_imagenet" "${CHECK_2}/resnet_small_normal_small_imagenet_seed.5_rf_level_8_recording_200_test_acc_49.5.pth" "_seed_5" 8 "normal" 1000 "${HOME}/features" "${HOME}/datasets" 0


# Level 9  Models

sbatch --nodes=1 --time=12:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="saving_features_lvl9_s_1.err" --output="saving_features_lvl9_s_1.out"  --job-name="saving_features_lvl9_s_1" slurm_similarity_run.sh "resnet_small" "small_imagenet" "${CHECK_2}/resnet_small_normal_small_imagenet_seed.5_rf_level_9_recording_200_test_acc_45.39.pth" "_seed_5" 9 "normal" 1000 "${HOME}/features" "${HOME}/datasets" 0
sbatch --nodes=1 --time=12:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="saving_features_lvl9_s_2.err" --output="saving_features_lvl9_s_2.out"  --job-name="saving_features_lvl9_s_2" slurm_similarity_run.sh "resnet_small" "small_imagenet" "${CHECK_2}/resnet_small_normal_small_imagenet_seed.6_rf_level_9_recording_200_test_acc_45.7.pth" "_seed_6" 9 "normal" 1000 "${HOME}/features" "${HOME}/datasets" 0

# Level 10  Models

sbatch --nodes=1 --time=12:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="saving_features_lvl10_s_1.err" --output="saving_features_lvl10_s_1.out"  --job-name="saving_features_lvl10_s_1" slurm_similarity_run.sh "resnet_small" "small_imagenet" "${CHECK_1}/resnet_small_normal_small_imagenet_seed.1_rf_level_10_recording_200_ffcv_test_acc_40.4.pth" "_seed_1" 10 "normal" 1000 "${HOME}/features" "${HOME}/datasets" 1
sbatch --nodes=1 --time=12:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="saving_features_lvl10_s_2.err" --output="saving_features_lvl10_s_2.out"  --job-name="saving_features_lvl10_s_2" slurm_similarity_run.sh "resnet_small" "small_imagenet" "${CHECK_1}/resnet_small_normal_small_imagenet_seed.2_rf_level_10_recording_200_ffcv_test_acc_39.8.pth" "_seed_2" 10 "normal" 1000 "${HOME}/features" "${HOME}/datasets" 1

# Similarity comparison for different receptive field levels

## Level 3  Models
#
#sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="comparison_lvl3_s1_V_s2_not_sub_mean.err" --output="comparison_lvl3_s1_V_s2_not_sub_mean.out"  --job-name="comparison_lvl3_s1_V_s2_not_sub_mean" slurm_similarity_comparison_run.sh "resnet_small" "small_imagenet" "_seed_1" "_seed_2" "normal" "normal" "npy" "npy" "${HOME}/features" 3 3 0
#sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="comparison_lvl3_s1_V_s2_sub_mean.err" --output="comparison_lvl3_s1_V_s2_sub_mean.out"  --job-name="comparison_lvl3_s1_V_s2_sub_mean" slurm_similarity_comparison_run.sh "resnet_small" "small_imagenet" "_seed_1" "_seed_2" "normal" "normal" "npy" "npy" "${HOME}/features" 3 3 1
#
## Level 4  Models
#
#sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="comparison_lvl4_s1_V_s2_not_sub_mean.err" --output="comparison_lvl4_s1_V_s2_not_sub_mean.out"  --job-name="comparison_lvl4_s1_V_s2_not_sub_mean" slurm_similarity_comparison_run.sh "resnet_small" "small_imagenet" "_seed_1" "_seed_2" "normal" "normal" "npy" "npy" "${HOME}/features" 4 4 0
#sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="comparison_lvl4_s1_V_s2_sub_mean.err" --output="comparison_lvl4_s1_V_s2_sub_mean.out"  --job-name="comparison_lvl4_s1_V_s2_sub_mean" slurm_similarity_comparison_run.sh "resnet_small" "small_imagenet" "_seed_1" "_seed_2" "normal" "normal" "npy" "npy" "${HOME}/features" 4 4 1
#
## Level 5  Models
#
#sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="comparison_lvl5_s1_V_s2_not_sub_mean.err" --output="comparison_lvl5_s1_V_s2_not_sub_mean.out"  --job-name="comparison_lvl5_s1_V_s2_not_sub_mean" slurm_similarity_comparison_run.sh "resnet_small" "small_imagenet" "_seed_5" "_seed_8" "normal" "normal" "npy" "npy" "${HOME}/features" 5 5 0
#sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="comparison_lvl5_s1_V_s2_sub_mean.err" --output="comparison_lvl5_s1_V_s2_sub_mean.out"  --job-name="comparison_lvl5_s1_V_s2_sub_mean" slurm_similarity_comparison_run.sh "resnet_small" "small_imagenet" "_seed_5" "_seed_8" "normal" "normal" "npy" "npy" "${HOME}/features" 5 5 1
#
## Level 6  Models
#
#sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="comparison_lvl6_s1_V_s2_not_sub_mean.err" --output="comparison_lvl6_s1_V_s2_not_sub_mean.out"  --job-name="comparison_lvl6_s1_V_s2_not_sub_mean" slurm_similarity_comparison_run.sh "resnet_small" "small_imagenet" "_seed_2" "_seed_3" "normal" "normal" "npy" "npy" "${HOME}/features" 6 6 0
#sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="comparison_lvl6_s1_V_s2_sub_mean.err" --output="comparison_lvl6_s1_V_s2_sub_mean.out"  --job-name="comparison_lvl6_s1_V_s2_sub_mean" slurm_similarity_comparison_run.sh "resnet_small" "small_imagenet" "_seed_2" "_seed_3" "normal" "normal" "npy" "npy" "${HOME}/features" 6 6 1
#
## Level 7  Models
#
#
#sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="comparison_lvl7_s1_V_s2_not_sub_mean.err" --output="comparison_lvl7_s1_V_s2_not_sub_mean.out"  --job-name="comparison_lvl7_s1_V_s2_not_sub_mean" slurm_similarity_comparison_run.sh "resnet_small" "small_imagenet" "_seed_3" "_seed_9" "normal" "normal" "npy" "npy" "${HOME}/features" 7 7 0
#sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="comparison_lvl7_s1_V_s2_sub_mean.err" --output="comparison_lvl7_s1_V_s2_sub_mean.out"  --job-name="comparison_lvl7_s1_V_s2_sub_mean" slurm_similarity_comparison_run.sh "resnet_small" "small_imagenet" "_seed_3" "_seed_9" "normal" "normal" "npy" "npy" "${HOME}/features" 7 7 1
#
## Level 8  Models
#
#sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="comparison_lvl8_s1_V_s2_not_sub_mean.err" --output="comparison_lvl8_s1_V_s2_not_sub_mean.out"  --job-name="comparison_lvl8_s1_V_s2_not_sub_mean" slurm_similarity_comparison_run.sh "resnet_small" "small_imagenet" "_seed_4" "_seed_5" "normal" "normal" "npy" "npy" "${HOME}/features" 8 8 0
#sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="comparison_lvl8_s1_V_s2_sub_mean.err" --output="comparison_lvl8_s1_V_s2_sub_mean.out"  --job-name="comparison_lvl8_s1_V_s2_sub_mean" slurm_similarity_comparison_run.sh "resnet_small" "small_imagenet" "_seed_4" "_seed_5" "normal" "normal" "npy" "npy" "${HOME}/features" 8 8 1
#
## Level 9  Models
#
#sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="comparison_lvl9_s1_V_s2_not_sub_mean.err" --output="comparison_lvl9_s1_V_s2_not_sub_mean.out"  --job-name="comparison_lvl9_s1_V_s2_not_sub_mean" slurm_similarity_comparison_run.sh "resnet_small" "small_imagenet" "_seed_5" "_seed_6" "normal" "normal" "npy" "npy" "${HOME}/features" 9 9 0
#sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="comparison_lvl9_s1_V_s2_sub_mean.err" --output="comparison_lvl9_s1_V_s2_sub_mean.out"  --job-name="comparison_lvl9_s1_V_s2_sub_mean" slurm_similarity_comparison_run.sh "resnet_small" "small_imagenet" "_seed_5" "_seed_6" "normal" "normal" "npy" "npy" "${HOME}/features" 9 9 1
#
#
## Level 10  Models
#
#sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="comparison_lvl10_s1_V_s2_not_sub_mean.err" --output="comparison_lvl10_s1_V_s2_not_sub_mean.out"  --job-name="comparison_lvl10_s1_V_s2_not_sub_mean" slurm_similarity_comparison_run.sh "resnet_small" "small_imagenet" "_seed_1" "_seed_2" "normal" "normal" "npy" "npy" "${HOME}/features" 10 10 0
#sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="comparison_lvl10_s1_V_s2_sub_mean.err" --output="comparison_lvl10_s1_V_s2_sub_mean.out"  --job-name="comparison_lvl10_s1_V_s2_sub_mean" slurm_similarity_comparison_run.sh "resnet_small" "small_imagenet" "_seed_1" "_seed_2" "normal" "normal" "npy" "npy" "${HOME}/features" 10 10 1

