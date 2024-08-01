CHECK_1="$HOME/checkpoints_temp"
CHECK_2="$HOME/checkpoints_arc4_2/checkpoints"


## Level 3  Models

#sbatch --nodes=1 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="logistic_probes_lvl3_s_1.err" --output="logistic_probes_lvl3_s_1.out"  --job-name="logistic_probes_lvl3_s_1" slurm_truncated_run.sh "resnet_small" "small_imagenet" "${CHECK_1}/resnet_small_normal_small_imagenet_seed.2_rf_level_3_recording_200_no_ffcv_test_acc_58.11.pth" "seed_2" 3 "normal" "${HOME}/truncated_models_results" 50 "0.1" 1

# Level 4  Model
sbatch --nodes=1 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="logistic_probes_lvl4_s_2.err" --output="logistic_probes_lvl4_s_2.out"  --job-name="logistic_probes_lvl4_s_2" slurm_truncated_run.sh "resnet_small" "small_imagenet" "${CHECK_1}/resnet_small_normal_small_imagenet_seed.2_rf_level_4_recording_200_ffcv_test_acc_60.31.pth" "seed_2" 4 "normal" "${HOME}/truncated_models_results" 50 "0.1" 1

# Level 5  Models

sbatch --nodes=1 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="logistic_probes_lvl5_s_1.err" --output="logistic_probes_lvl5_s_1.out"  --job-name="logistic_probes_lvl5_s_1" slurm_truncated_run.sh "resnet_small" "small_imagenet" "${CHECK_2}/resnet_small_normal_small_imagenet_seed.5_rf_level_5_recording_200_test_acc_61.97.pth" "seed_5" 5 "normal" "${HOME}/truncated_models_results" 50 "0.1" 0

# Level 6  Models

sbatch --nodes=1 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="logistic_probes_lvl6_s_1.err" --output="logistic_probes_lvl6_s_1.out"  --job-name="logistic_probes_lvl6_s_1" slurm_truncated_run.sh "resnet_small" "small_imagenet" "${CHECK_1}/resnet_small_normal_small_imagenet_seed.3_rf_level_6_recording_200_ffcv_test_acc_57.83.pth" "seed_3" 6 "normal" "${HOME}/truncated_models_results" 50 "0.1" 1

# Level 7  Models

sbatch --nodes=1 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="logistic_probes_lvl7_s_1.err" --output="logistic_probes_lvl7_s_1.out"  --job-name="logistic_probes_lvl7_s_1" slurm_truncated_run.sh "resnet_small" "small_imagenet" "${CHECK_2}/resnet_small_normal_small_imagenet_seed.9_rf_level_7_recording_200_test_acc_51.51.pth" "seed_9" 7 "normal" "${HOME}/truncated_models_results" 50 "0.1" 0

# Level 8  Models

sbatch --nodes=1 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="logistic_probes_lvl8_s_1.err" --output="logistic_probes_lvl8_s_1.out"  --job-name="logistic_probes_lvl8_s_1" slurm_truncated_run.sh "resnet_small" "small_imagenet" "${CHECK_2}/resnet_small_normal_small_imagenet_seed.4_rf_level_8_recording_200_test_acc_49.1.pth" "seed_4" 8 "normal" "${HOME}/truncated_models_results" 50 "0.1" 0


# Level 9  Models

sbatch --nodes=1 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="logistic_probes_lvl9_s_1.err" --output="logistic_probes_lvl9_s_1.out"  --job-name="logistic_probes_lvl9_s_1" slurm_truncated_run.sh "resnet_small" "small_imagenet" "${CHECK_2}/resnet_small_normal_small_imagenet_seed.5_rf_level_9_recording_200_test_acc_45.39.pth" "seed_5" 9 "normal" "${HOME}/truncated_models_results" 50 "0.1" 0

# Level 10  Models

sbatch --nodes=1 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="logistic_probes_lvl10_s_1.err" --output="logistic_probes_lvl10_s_1.out"  --job-name="logistic_probes_lvl10_s_1" slurm_truncated_run.sh "resnet_small" "small_imagenet" "${CHECK_1}/resnet_small_normal_small_imagenet_seed.1_rf_level_10_recording_200_ffcv_test_acc_40.4.pth" "seed_1" 10 "normal" "${HOME}/truncated_models_results" 50 "0.01" 1
