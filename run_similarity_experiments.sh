#!/bin/bash -l
# Submission script for serial Python job
# Luis Alfredo Avendano  2021-10-22

# Run from the current directory and with current environment
#$ -cwd -V

# Ask for some time (hh:mm:ss max of 48:00:00)
#$ -l h_rt=03:00:00

# ASk for some GPU
#$ -l coproc_p100=1

# Ask for some memory (by default, 1G, without a request)
#$ -l h_vmem=16G
# -pe smp 3
# Send emails when job starts and ends
#$ -m e
######## FOR RESNET50       ###################################################################
#
## Level 0
#rf_level0_s1="trained_models/cifar10/resnet50_cifar10.pth"
#name_rf_level0_s1="_seed_1_rf_level_0"
#
#
#rf_level0_s2="trained_models/cifar10/resnet50_normal_seed_2_tst_acc_95.65.pth"
#name_rf_level0_s2="_seed_2_rf_level_0"
#
## level 1
#rf_level1_s1="trained_models/cifar10/resnet50_normal_cifar10_seed_1_rf_level_1_95.26.pth"
#name_rf_level1_s1="_seed_1_rf_level_1"
#rf_level1_s2="trained_models/cifar10/resnet50_normal_cifar10_seed_2_rf_level_1_94.93.pth"
#name_rf_level1_s2="_seed_2_rf_level_1"
#
## level 2
#rf_level2_s1="trained_models/cifar10/resnet50_normal_cifar10_seed_1_rf_level_2_94.07.pth"
#name_rf_level2_s1="_seed_1_rf_level_2"
#
#rf_level2_s2="trained_models/cifar10/resnet50_normal_cifar10_seed_2_rf_level_2_94.03.pth"
#name_rf_level2_s2="_seed_2_rf_level_2"
##Level 3
#
#rf_level3_s1="trained_models/cifar10/resnet50_normal_cifar10_seed_1_rf_level_3_92.38.pth"
#name_rf_level3_s1="_seed_1_rf_level_3"
#
#rf_level3_s2="trained_models/cifar10/resnet50_normal_cifar10_seed_2_rf_level_3_92.25.pth"
#name_rf_level3_s2="_seed_2_rf_level_3"
#
#
##Level 4
#rf_level4_s1="trained_models/cifar10/resnet50_normal_cifar10_seed_1_rf_level_4_90.66.pth"
#name_rf_level4_s1="_seed_1_rf_level_4"
#rf_level4_s2="trained_models/cifar10/resnet50_normal_cifar10_seed_2_rf_level_4_90.8.pth"
#name_rf_level4_s2="_seed_2_rf_level_4"
#
## Pytorch implementation
#
#rf_level_p_s1="trained_models/cifar10/resnet50_official_cifar10_seed_1_test_acc_90.31.pth"
#name_rf_level_p_s1="_seed_1_rf_level_p"
#
#rf_level_p_s2="trained_models/cifar10/resnet50_official_cifar10_seed_2_test_acc_89.93.pth"
#name_rf_level_p_s2="_seed_2_rf_level_p"
#
#rf_level_p_s3="trained_models/cifar10/resnet50_pytorch_cifar10_seed_3_test_acc_89.33.pth"
#name_rf_level_p_s3="_seed_3_rf_level_p"
#
#datasets
#models
#declare -A pruning_rates
#declare -A sigmas
#pruning_rates()
#
#
#max1=${#files[@]}                                  # Take the length of that array
#max2=${#files[@]}                                  # Take the length of that array
#for ((idxA=0; idxA<max; idxA++)); do              # iterate idxA from 0 to length
#  for ((idxB=idxA+1; idxB<max; idxB++)); do         # iterate idxB from idxA to length
#qsub -N "fine_tune_stochastic pruning_${}_${}_${}__${}_${}" run.sh
#done
#done

#                         RESUME Training of  specific seeds
# VGG small imagenet lvl 3
#qsub -l h_rt=45:00:00 -l coproc_v100=1  -N "training_Level_3_vgg_small_imagenet" resume_run.sh "vgg19" "small_imagenet" 2 3 "normal" 200 "recording_200" 1 1 "/nobackup/sclaam/checkpoints/vgg19_normal_small_imagenet_seed.1_rf_level_3_recording_200_test_acc_57.65.pth"
# VGG small imagenet lvl 4
#qsub -l h_rt=45:00:00 -l coproc_v100=1  -N "training_Level_4_vgg_small_imagenet" resume_run.sh "vgg19" "small_imagenet" 2 4 "normal" 200 "recording_200" 1 1 "/nobackup/sclaam/checkpoints/vgg19_normal_small_imagenet_seed.1_rf_level_4_recording_200_test_acc_55.02.pth"
#
## VGG small imagenet lvl 2 full training
#qsub -l h_rt=45:00:00 -l coproc_v100=1 -N "training_Level_2_vgg_small_imagenet" run.sh "vgg19" "small_imagenet" 2 2 "normal" 200 "recording_200" 1 1

###                       resnet18
### resnet18 small imagenet lvl 2
#qsub -l h_rt=45:00:00 -l coproc_v100=1 -N "resume_training_Level_2_renet18_small_imagenet" resume_run.sh "resnet18" "small_imagenet" 2 2 "normal" 200 "recording_200" 1 1 "/nobackup/sclaam/checkpoints/resnet18_normal_small_imagenet_seed.0_rf_level_2_recording_200_test_acc_64.99.pth"
### resnet18 small imagenet lvl 3
#qsub -l h_rt=45:00:00 -l coproc_v100=1 -N "resume_training_Level_3_renet18_small_imagenet" resume_run.sh "resnet18" "small_imagenet" 2 3 "normal" 200 "recording_200" 1 1 "/nobackup/sclaam/checkpoints/resnet18_normal_small_imagenet_seed.0_rf_level_3_recording_200_test_acc_71.05.pth"
## resnet18 small imagenet lvl 4
#qsub -l h_rt=45:00:00 -l coproc_v100=1 -N "resume_training_Level_4_renet18_small_imagenet" resume_run.sh "resnet18" "small_imagenet" 2 4 "normal" 200 "recording_200" 1 1 "/nobackup/sclaam/checkpoints/resnet18_normal_small_imagenet_seed.0_rf_level_4_recording_200_test_acc_69.19.pth"
### resnet18 small imagenet lvl K6
#qsub -l h_rt=45:00:00 -l coproc_v100=1 -N "resume_training_Level_k6_renet18_small_imagenet" resume_run.sh "resnet18" "small_imagenet" 2 "k6" "normal" 200 "recording_200" 1 1 "/nobackup/sclaam/checkpoints/resnet18_normal_small_imagenet_1715866976.78545_rf_level_k6_recording_200_test_acc_77.0.pth"
### resnet18 small imagenet lvl K7
#qsub -l h_rt=45:00:00 -l coproc_v100=1 -N "resume_training_Level_k7_renet18_small_imagenet" resume_run.sh "resnet18" "small_imagenet" 2 "k7" "normal" 200 "recording_200" 1 1 "/nobackup/sclaam/checkpoints/resnet18_normal_small_imagenet_1715893994.3666036_rf_level_k7_recording_200_test_acc_75.44.pth"

### resnet18 small imagenet lvl K8
#qsub -l h_rt=48:00:00 -l coproc_v100=1 -N "resume_training_Level_k8_renet18_small_imagenet" resume_run.sh "resnet18" "small_imagenet" 2 "k8" "normal" 200 "recording_200" 1 1 "/nobackup/sclaam/checkpoints/resnet18_normal_small_imagenet_1715962172.5830548_rf_level_k8_recording_200_test_acc_70.75.pth"
### resnet18 small imagenet lvl K9
#qsub -l h_rt=48:00:00 -l coproc_v100=1 -N "resume_training_Level_k9_renet18_small_imagenet" resume_run.sh "resnet18" "small_imagenet" 2 "k9" "normal" 200 "recording_200" 1 1 "/nobackup/sclaam/checkpoints/resnet18_normal_small_imagenet_1716016178.5822153_rf_level_k9_recording_200_test_acc_50.14.pth"
#
#qsub -l h_rt=48:00:00 -l coproc_v100=1 -N "resume_training_Level_3_small_resnet_small_imagenet" resume_run.sh "resnet_small" "small_imagenet" 2 3 "normal" 200 "recording_200" 1 1 "/nobackup/sclaam/checkpoints/resnet_small_normal_small_imagenet_seed.0_rf_level_3_recording_200_test_acc_42.89.pth"

### resnet18 small imagenet lvl K9
#qsub -l h_rt=48:00:00 -l coproc_v100=1 -N "resume_training_Level_5_small_resnet_small_imagenet" resume_run.sh "resnet_small" "small_imagenet" 2 5 "normal" 200 "recording_200" 1 1 "/nobackup/sclaam/checkpoints/resnet_small_normal_small_imagenet_seed.0_rf_level_5_recording_200_test_acc_46.75.pth"
#
#qsub -l h_rt=48:00:00 -l coproc_v100=1 -N "resume_training_Level_7_small_resnet_small_imagenet" resume_run.sh "resnet_small" "small_imagenet" 2 7 "normal" 200 "recording_200" 1 1 "/nobackup/sclaam/checkpoints/resnet_small_normal_small_imagenet_seed.0_rf_level_7_recording_200_test_acc_43.93.pth"



#
#qsub -l h_rt=6:00:00 -l coproc_v100=1 -N "Kfac_optim_rf_1_cifar10_rs" run.sh "cifar10" "resnet50" "1" "normal" "kfac_optim_hyper" "1"
#qsub -l h_rt=6:00:00 -l coproc_v100=1 -N "Kfac_optim_rf_2_cifar10_rs" run.sh "cifar10" "resnet50" "2" "normal" "kfac_optim_hyper" "1"
#qsub -l h_rt=6:00:00 -l coproc_v100=1 -N "Kfac_optim_rf_3_cifar10_rs" run.sh "cifar10" "resnet50" "3" "normal" "kfac_optim_hyper" "1"
#qsub -l h_rt=6:00:00 -l coproc_v100=1 -N "Kfac_optim_rf_4_cifar10_rs" run.sh "cifar10" "resnet50" "4" "normal" "kfac_optim_hyper" "1"
#qsub -l h_rt=6:00:00 -l coproc_v100=1 -N "Kfac_optim_rf_k6_cifar10_rs" run.sh "cifar10" "resnet50" "k6" "normal" "kfac_optim_hyper" "1"
#qsub -l h_rt=6:00:00 -l coproc_v100=1 -N "Kfac_optim_rf_k7_cifar10_rs" run.sh "cifar10" "resnet50" "k7" "normal" "kfac_optim_hyper" "1"
#qsub -l h_rt=6:00:00 -l coproc_v100=1 -N "Kfac_optim_rf_k8_cifar10_rs" run.sh "cifar10" "resnet50" "k8" "normal" "kfac_optim_hyper" "1"
#qsub -l h_rt=6:00:00 -l coproc_v100=1 -N "Kfac_optim_rf_k9_cifar10_rs" run.sh "cifar10" "resnet50" "k9" "normal" "kfac_optim_hyper" "1"



#qsub -l h_rt=30:00:00 -l coproc_v100=1 -N "sam_hyper_parameter_Optim" run.sh "sam"
#qsub -l h_rt=30:00:00 -l coproc_v100=1 -N "kfac_hyper_parameter_Optim" run.sh "kfac"

#qsub -l h_rt=6:00:00 -l coproc_v100=1 -N "sam_optim_rf_1_cifar10_rs" run.sh "cifar10" "resnet50" "1" "normal" "sam_optim_hyper" "2"
#qsub -l h_rt=6:00:00 -l coproc_v100=1 -N "sam_optim_rf_2_cifar10_rs" run.sh "cifar10" "resnet50" "2" "normal" "sam_optim_hyper" "2"
#qsub -l h_rt=6:00:00 -l coproc_v100=1 -N "sam_optim_rf_3_cifar10_rs" run.sh "cifar10" "resnet50" "3" "normal" "sam_optim_hyper" "2"
#qsub -l h_rt=6:00:00 -l coproc_v100=1 -N "sam_optim_rf_4_cifar10_rs" run.sh "cifar10" "resnet50" "4" "normal" "sam_optim_hyper" "2"
#qsub -l h_rt=6:00:00 -l coproc_v100=1 -N "sam_optim_rf_k6_cifar10_rs" run.sh "cifar10" "resnet50" "k6" "normal" "sam_optim_hyper" "2"
#qsub -l h_rt=6:00:00 -l coproc_v100=1 -N "sam_optim_rf_k7_cifar10_rs" run.sh "cifar10" "resnet50" "k7" "normal" "sam_optim_hyper" "2"
#qsub -l h_rt=6:00:00 -l coproc_v100=1 -N "sam_optim_rf_k8_cifar10_rs" run.sh "cifar10" "resnet50" "k8" "normal" "sam_optim_hyper" "2"
#qsub -l h_rt=6:00:00 -l coproc_v100=1 -N "sam_optim_rf_k9_cifar10_rs" run.sh "cifar10" "resnet50" "k9" "normal" "sam_optim_hyper" "2"

#
#qsub -l h_rt=44:00:00 -l coproc_k80=1 -N "Kfac_optim_rf_2_tiny_imagenet" run.sh "tiny_imagenet" "resnet50" "2" "normal" "kfac_optimiser_2"
#qsub -l h_rt=44:00:00 -l coproc_k80=1 -N "Kfac_optim_rf_3_tiny_imagenet" run.sh "tiny_imagenet" "resnet50" "3" "normal" "kfac_optimiser_2"
#qsub -l h_rt=44:00:00 -l coproc_k80=1 -N "Kfac_optim_rf_4_tiny_imagenet" run.sh "tiny_imagenet" "resnet50" "4" "normal" "kfac_optimiser_2"
#qsub -l h_rt=44:00:00 -l coproc_p100=1 -N "Kfac_optim_rf_5_tiny_imagenet" run.sh "tiny_imagenet" "resnet50" "5" "normal" "kfac_optimiser_2"
#qsub -l h_rt=44:00:00 -l coproc_p100=1 -N "Kfac_optim_rf_6_tiny_imagenet" run.sh "tiny_imagenet" "resnet50" "6" "normal" "kfac_optimiser_2"
#qsub -l h_rt=44:00:00 -l coproc_k80=1 -N "Kfac_optim_rf_7_tiny_imagenet" run.sh "tiny_imagenet" "resnet50" "7" "normal" "kfac_optimiser_2"


#qsub -l h_rt=45:00:00 -l coproc_k80=1  -N "training_Level_1_vgg_small_imagenet" run.sh "vgg19" "small_imagenet" 2 1 "normal" 200 "recording_200" 1 1
#qsub -l h_rt=45:00:00 -l coproc_v100=1  -N "training_Level_k6_vgg_small_imagenet" run.sh "vgg19" "small_imagenet" 4 "k6" "normal" 200 "recording_200" 1 1
#qsub -l h_rt=45:00:00 -l coproc_v100=1  -N "training_Level_k7_vgg_small_imagenet" run.sh "vgg19" "small_imagenet" 4 "k7" "normal" 200 "recording_200" 1 1
#qsub -l h_rt=45:00:00 -l coproc_v100=1  -N "training_Level_k8_vgg_small_imagenet" run.sh "vgg19" "small_imagenet" 4 "k8" "normal" 200 "recording_200" 1 1
#qsub -l h_rt=45:00:00 -l coproc_v100=1  -N "training_Level_k9_vgg_small_imagenet" run.sh "vgg19" "small_imagenet" 4 "k9" "normal" 200 "recording_200" 1 1

#qsub -l h_rt=45:00:00 -l coproc_v100=1  -N "training_Level_3_vgg_small_imagenet" run.sh "vgg19" "small_imagenet" 2 3 "normal" 200 "recording_200" 1 1
#qsub -l h_rt=12:00:00 -l coproc_p100=1  -N "training_Level_3_vgg_small_cifar10" run.sh "vgg_small" "small_imagenet" 2 3 "normal" 200 "recording" 1 1
#qsub -l h_rt=45:00:00 -l coproc_v100=1  -N "training_Level_4_vgg_small_imagenet" run.sh "vgg19" "small_imagenet" 2 4 "normal" 200 "recording_200" 1 1
#
#qsub -l h_rt=45:00:00 -l coproc_v100=1  -N "training_Level_1_resnet18_small_imagenet" run.sh "resnet18" "small_imagenet" 4 1 "normal" 200 "recording_200" 1 1

#qsub -l h_rt=45:00:00 -l coproc_v100=1  -N "training_Level_k6_resnet18_small_imagenet" run.sh "resnet18" "small_imagenet" 4 "k6" "normal" 200 "recording_200" 1 1
#qsub -l h_rt=45:00:00 -l coproc_v100=1  -N "training_Level_k7_resnet18_small_imagenet" run.sh "resnet18" "small_imagenet" 4 "k7" "normal" 200 "recording_200" 1 1
#qsub -l h_rt=45:00:00 -l coproc_v100=1  -N "training_Level_k8_resnet18_small_imagenet" run.sh "resnet18" "small_imagenet" 4 "k8" "normal" 200 "recording_200" 1 1
#qsub -l h_rt=45:00:00 -l coproc_v100=1  -N "training_Level_k9_resnet18_small_imagenet" run.sh "resnet18" "small_imagenet" 4 "k9" "normal" 200 "recording_200" 1 1

#qsub -l h_rt=45:00:00 -l coproc_v100=1  -N "training_Level_3_resnet18_small_imagenet" run.sh "resnet18" "small_imagenet" 4 3 "normal" 200 "recording_200" 1 1
#qsub -l h_rt=45:00:00 -l coproc_v100=1  -N "training_Level_4_resnet18_small_imagenet" run.sh "resnet18" "small_imagenet" 4 4 "normal" 200 "recording_200" 1 1
#qsub -l h_rt=45:00:00 -l coproc_v100=1  -N "training_Level_5_resnet18_small_imagenet" run.sh "resnet18" "small_imagenet" 4 5 "normal" 200 "recording_200" 1 1
#qsub -l h_rt=45:00:00 -l coproc_v100=1  -N "training_Level_6_resnet18_small_imagenet" run.sh "resnet18" "small_imagenet" 4 6 "normal" 200 "recording_200" 1 1
#qsub -l h_rt=12:00:00 -l coproc_v100=1  -N "training_Level_6_resnet_small_imagenet" run.sh "resnet_small" "small_imagenet" 1 6 "normal" 200 "recording" 1 1
#qsub -l h_rt=6:00:00 -l coproc_p100=1  -N "training_Level_7_resnet_small_cifar10" run.sh "resnet_small" "cifar10" 2 7 "normal" 400 "recording_400" 1 1

#qsub -l h_rt=45:00:00 -l coproc_v100=1  -N "training_Level_7_resnet18_small_imagenet" run.sh "resnet18" "small_imagenet" 2 7 "normal" 200 "recording_200" 1 1
# aquí estoy
#qsub -l h_rt=30:00:00 -l coproc_p100=1 -N "hyper_parameter_optim_second_order" run.sh

#type="one_shot"

#qsub -l h_rt=6:00:00 -l coproc_p100=1 -N "linear_1_interpolation_experiments" run.sh 0
#qsub -l h_rt=6:00:00 -l coproc_p100=1 -N "linear_2_interpolation_experiments" run.sh 1
#qsub -l h_rt=6:00:00 -l coproc_p100=1 -N "linear_3_interpolation_experiments" run.sh 2
#qsub -l h_rt=6:00:00 -l coproc_p100=1 -N "linear_4_interpolation_experiments" run.sh 3
#qsub -l h_rt=6:00:00 -l coproc_p100=1 -N "linear_5_interpolation_experiments" run.sh 4
#qsub -l h_rt=6:00:00 -l coproc_p100=1 -N "linear_6_interpolation_experiments" run.sh 5

#type="dense"

#qsub -l h_rt=6:00:00 -l coproc_p100=1 -N "fine_tune_stochastic_pruning_cifar10_resnet18_0.005_0.9" run.sh "11" "0.005" "global" "resnet18" "cifar10" "0.9" "alternative" "200"
#qsub -l h_rt=6:00:00 -l coproc_p100=1 -N "fine_tune_stochastic_pruning_cifar10_resnet50_0.003_0.95" run.sh "11" "0.003" "global" "resnet50" "cifar10" "0.95" "alternative" "200"
#qsub -l h_rt=6:00:00 -l coproc_p100=1 -N "fine_tune_stochastic_pruning_cifar10_vgg19_0.003_0.95" run.sh "11" "0.003" "global" "VGG19" "cifar10" "0.95" "alternative" "200"
#qsub -l h_rt=6:00:00 -l coproc_p100=1 -N "fine_tune_stochastic_pruning_cifar100_resnet18_0.003_0.9" run.sh "11" "0.003" "global" "resnet18" "cifar100" "0.9" "alternative" "200"
#qsub -l h_rt=6:00:00 -l coproc_p100=1 -N "fine_tune_stochastic_pruning_cifar100_resnet50_0.001_0.85" run.sh "11" "0.001" "global" "resnet50" "cifar100" "0.85" "alternative" "200"
#qsub -l h_rt=6:00:00 -l coproc_p100=1 -N "fine_tune_stochastic_pruning_cifar100_vgg19_0.001_0.8" run.sh "11" "0.001" "global" "VGG19" "cifar100" "0.8" "alternative" "200"
##python stochastic_loss_landscape.py --sigma "0.001" --batch_size 128 -pr "0.8" -dt "cifar100" -ar "vgg19" -mt "alternative"
#python stochastic_loss_landscape.py --sigma "0.001" --batch_size 128 --pruner "global" -pru "0.8" -dt "cifar100" -ar "VGG19" -mt "alternative" -id "test"
#python stochastic_loss_landscape.py --sigma "0.001" --batch_size 128 --pruner "global" -pru "0.8" -dt "cifar100" -ar "VGG19" -mt "alternative" -id "test"
#python stochastic_loss_landscape.py --sigma "0.001" --batch_size 128 --pruner "global" -pru "0.8" -dt "cifar100" -ar "VGG19" -mt "alternative" -id "test"
#python stochastic_loss_landscape.py --sigma "0.001" --batch_size 128 --pruner "global" -pru "0.8" -dt "cifar100" -ar "VGG19" -mt "alternative" -id "test"
#python stochastic_loss_landscape.py --sigma "0.001" --batch_size 128 --pruner "global" -pru "0.8" -dt "cifar100" -ar "VGG19" -mt "alternative" -id "test"
#python stochastic_loss_landscape.py --sigma "0.001" --batch_size 128 --pruner "global" -pru "0.8" -dt "cifar100" -ar "VGG19" -mt "alternative" -id "test"
#
#python stochastic_loss_landscape --sigma "0.001" --batch_size 128 --pruner "global" -pru "0.8" -dt "cifar100" -ar "VGG19" -mt "alternative" -id "test" -nw 2

#python main.py -exp 18 -bs 128 --sigma $2 --pruner $3 --architecture $4 --dataset $5 --pruning_rate $6 --modeltype $7 --epochs $8

#qsub -N "saving_features_resnet50_rf_levelp_s3_pytorch_no_train" run.sh  "resnet50" "filer" "pytorch_no_train_1" 0 hub
#qsub -N "saving_features_resnet50_rf_levelp_s3_pytorch_no_train" run.sh  "resnet50" "filer" "pytorch_no_train_2" 0 hub
#qsub -N "saving_features_resnet50_rf_level0_s3_pytorch_no_train" run.sh  "resnet50" "filer" "custom_no_train_1" 0 alternative
#qsub -N "saving_features_resnet50_rf_level0_s3_pytorch_no_train" run.sh  "resnet50" "filer" "custom_no_train_2" 0 alternative

#./run.sh  "resnet50" foreing_trained_models/cifar10/resnet50_pytorch_cifar10_seed_3_test_acc_89.33.pth "${name_rf_level_p_s3}" 0 hub

#qsub -N "saving_features_resnet50_rf_level1_s1" run.sh  "resnet50" "${rf_level1_s1}" "${name_rf_level1_s1}" 1 alternative
#qsub -N "saving_features_resnet50_rf_level1_s2" run.sh  "resnet50" "${rf_level1_s2}" "${name_rf_level1_s2}" 1 alternative
#qsub -N "saving_features_resnet50_rf_level3_s1" run.sh  "resnet50" "${rf_level3_s1}" "${name_rf_level3_s1}" 3 alternative
#qsub -N "saving_features_resnet50_rf_level3_s2" run.sh  "resnet50" "${rf_level3_s2}" "${name_rf_level3_s2}" 3 alternative
#qsub -N "saving_features_resnet50_rf_level2_s1" run.sh  "resnet50" "${rf_level2_s1}" "${name_rf_level2_s1}" 2 alternative
#qsub -N "saving_features_resnet50_rf_level2_s2" run.sh  "resnet50" "${rf_level2_s2}" "${name_rf_level2_s2}" 2 alternative
#qsub -N "saving_features_resnet50_rf_level4_s1" run.sh  "resnet50" "${rf_level4_s1}" "${name_rf_level4_s1}" 4 alternative
#qsub -N "saving_features_resnet50_rf_level4_s2" run.sh  "resnet50" "${rf_level4_s2}" "${name_rf_level4_s2}" 4 alternative

#
#files=($name_rf_level1_s1 $name_rf_level1_s2 $name_rf_level2_s1 $name_rf_level2_s2 $name_rf_level3_s1 $name_rf_level3_s2 $name_rf_level4_s1 $name_rf_level4_s2)
#max=${#files[@]}                                  # Take the length of that array
#for ((idxA=0; idxA<max; idxA++)); do              # iterate idxA from 0 to length
#  for ((idxB=idxA+1; idxB<max; idxB++)); do         # iterate idxB from idxA to length
#    echo "A: ${files[$idxA]}; B: ${files[$idxB]}" # Do whatever you're here for.
#    qsub -N "similarity_${files[$idxA]}_${files[$idxB]}" run.sh  "resnet50" "${files[$idxA]}" "${files[$idxB]}" "alternative" "alternative" "npy" "npy"
#  done
#done

#qsub -N "similarity_no_train_1" run.sh  "resnet50" "pytorch_no_train_1" "pytorch_no_train_2" "hub" "hub" "npy" "npy"
#qsub -N "similarity_no_train_2" run.sh  "resnet50" "custom_no_train_1" "custom_no_train_2" "alternative" "alternative" "npy" "npy"
#
#qsub -N "similarity_tird_seed_pytorch_1_k80" run.sh  "resnet50"  "${name_rf_level_p_s3}" "_seed_2" "${rf_level_p_s3}"
#./run.sh  "resnet50"  "${name_rf_level_p_s3}" "_seed_2" "${rf_level_p_s3}"



###############################################################################################
#                                              Smoothness
###############################################################################################
#files_names=($name_rf_level0_s1 $name_rf_level1_s1  $name_rf_level2_s1 $name_rf_level3_s1 $name_rf_level4_s1 $name_rf_level_p_s1)
#files=($rf_level0_s1 $rf_level1_s1  $rf_level2_s1 $rf_level3_s1 $rf_level4_s1 $rf_level_p_s1)
#files_level=(0 1 2 3 4 0)
#files_type=("normal" "normal" "normal" "normal" "normal" "pytorch")
#max=${#files[@]}                                  # Take the length of that array
#for ((idxA=0; idxA<max; idxA++)); do              # iterate idxA from 0 to length
#qsub -N "loading_model_test_${files_names[$idxA]}" run.sh  "resnet50" "cifar10" "${files_level[$idxA]}" "${files_type[$idxA]}" "${files_names[$idxA]}" "${files[$idxA]}"
#echo "loading_model_test_${files_names[$idxA]}"
#./run.sh  "resnet50" "cifar10" "${files_level[$idxA]}" "${files_type[$idxA]}" "${files_names[$idxA]}" "${files[$idxA]}"
#done
#
#files_names=($name_rf_level0_s2 $name_rf_level1_s2  $name_rf_level2_s2 $name_rf_level3_s2 $name_rf_level4_s2 $name_rf_level_p_s2)
#files=($rf_level0_s2 $rf_level1_s2  $rf_level2_s2 $rf_level3_s2 $rf_level4_s2 $rf_level_p_s2)

#for ((idxA=0; idxA<max; idxA++)); do              # iterate idxA from 0 to length
#      qsub -N "${files_names[$idxA]}" run.sh  "resnet50" "cifar10" "${files_level[$idxA]}" "${files_type[$idxA]}" "${files_names[$idxA]}" "${files[$idxA]}"
#      echo "loading_model_test_${fles_names[$idxA]}"
#      ./run.sh  "resnet50" "cifar10" "${files_level[$idxA]}" "${files_type[$idxA]}" "${files_names[$idxA]}" "${files[$idxA]}"
#done

#models=("resnet50")
##Datasets
#datasets=("cifar10" "cifar100")
#types=("alternative")
#rf_levels=("2")
########################################################################################################################
#                   Training a model on a given dataset with a given receptive field
########################################################################################################################

#qsub -N "training_Level_0_vgg" run.sh "vgg19" "tiny_imagenet" 2 0 "normal" 300
#qsub -N "training_Level_1_vgg" run.sh "vgg19" "tiny_imagenet" 2 1 "normal" 300
#qsub -N "training_Level_2_vgg" run.sh "vgg19" "tiny_imagenet" 2 2 "normal" 300
#qsub -N "training_Level_3_vgg" run.sh "vgg19" "tiny_imagenet" 2 3 "normal" 300
#qsub -N "training_Level_4_vgg" run.sh "vgg19" "tiny_imagenet" 2 4 "normal" 300

#qsub  -l coproc_p100=1  -N "train_cf10_Level_2_rs_record" run.sh "resnet50" "cifar10" 8 2 "normal" 200 "recording" 1 1

#qsub -l h_rt=6:00:00 -t 1-5 -l coproc_p100=1  -N "train_cifar10_Level_0_r50_record_depth_experiment" run.sh "vgg19" "cifar10" 2 0 "normal" 200 "no_recording" 1 0
            #             this did not happened
#qsub -l h_rt=6:00:00 -t 1-5 -l coproc_p100=1  -N "train_cifar10_Level_0_r50_record_depth_experiment" run.sh "resnet50" "cifar10" 2 0 "normal" 200 "no_recording" 1 0

#qsub -l h_rt=6:00:00 -t 1-5 -l coproc_p100=1  -N "train_tinm_Level_0_r50_record_depth_experiment" run.sh "resnet50" "tiny_imagenet" 2 0 "normal" 200 "no_recording_bs_32" 1 0
#qsub -l h_rt=6:00:00 -t 1-5 -l coproc_p100=1  -N "train_tinm_Level_1_r50_record_depth_experiment" run.sh "resnet50" "tiny_imagenet" 2 1 "normal" 200 "no_recording_bs_32" 1 0
#qsub -l h_rt=6:00:00 -t 1-5 -l coproc_p100=1  -N "train_tnim_Level_0_50_record_depth_experiment" run.sh "resnet50" "tiny_imagenet" 2 0 "normal" 200 "no_recording" 1 0

#qsub  -l coproc_p100=1  -N "train_tnim_Level_3_rs_no_record_width_2" run.sh "resnet50" "tiny_imagenet" 4 3 "normal" 200 "no_recording_width_2" 2 0
#qsub  -l coproc_p100=1  -N "train_tnim_Level_3_rs_no_record_width_3_finish" run.sh "resnet50" "tiny_imagenet" 8 3 "normal" 200 "no_recording_width_3_second_attemp" 3 0
#
#qsub  -l coproc_p100=1  -N "train_tnim_Level_5_rs_no_record_width_2" run.sh "resnet50" "tiny_imagenet" 4 5 "normal" 200 "no_recording_width_2" 2 0
#qsub  -l coproc_p100=1  -N "train_tnim_Level_5_rs_no_record_width_3" run.sh "resnet50" "tiny_imagenet" 4 5 "normal" 200 "no_recording_width_3" 3 0

#qsub  -l coproc_p100=1  -N "train_tnim_Level_7_rs_record" run.sh "resnet50" "tiny_imagenet" 8 7 "normal" 200 "recording" 1 1
#qsub  -l coproc_k80=1 -t 1-5  -N "train_Level_5_rs_" run.sh "resnet50" "cifar10" 2 5 "normal" 200 "no_recording" 1 0
#qsub  -l coproc_k80=1 -t 1-5 -N "training_Level_6_rs" run.sh "resnet50" "cifar10" 2 6 "normal" 200 "no_recording" 1 0
#qsub -l coproc_k80=1 -t 1-5 -N "training_Level_7_rs" run.sh "resnet50" "cifar10" 2 7 "normal" 200 "no_recording" 1 0
# -l coproc_p100=1
#qsub -l coproc_p100=1 -t 1-5 -N "TL_4_BS_64_rs" run.sh "resnet50" "tiny_imagenet" 8 4 "normal" 300 "bs_32"
#qsub -l coproc_p100=1 -t 1-5 -N "TL_2_BS_64_rs" run.sh "resnet50" "tiny_imagenet" 8 2 "normal" 300 "bs_32"
#qsub -l coproc_k80=1 -t 4-5 -N  "training_Level_1_rs.2" run.sh "resnet50" "tiny_imagenet" 8 1 "normal" 300
#qsub -N "training_Level_2_rs" run.sh "resnet50" "tiny_imagenet" 2 2 "normal" 300
#qsub -N "training_Level_3_rs" run.sh "resnet50" "tiny_imagenet" 2 3 "normal" 300
#qsub -N "training_Level_4_rs" run.sh "resnet50" "tiny_imagenet" 2 4 "normal" 300

########################################################################################################################


#     initial weights feature representation
#####################################################
#             Logistic features
#####################################################

#model="resnet50"
#dataset="cifar10"
#init=0
#experiment=1
##solution_string="initial_weights"
#solution_string="no_recording_test_acc"
#directory=/nobackup/sclaam/checkpoints
##directory=trained_models
#level_1_files=($(ls $directory | grep -i "${model}.*${dataset}.*_level_1_.*${solution_string}.*"))
#level_1_files=($(ls $directory | grep -i "${model}.*${dataset}.*_level_1.pth"))
#
##level_5_files=($(ls $directory | grep -i "${model}.*${dataset}.*_level_4_.*${solution_string}.*"))
#level_4_files=($(ls $directory | grep -i "${model}.*${dataset}.*_level_4.pth"))
##level_2_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_2.pth"))
#level_7_files=($(ls $directory | grep -i "${model}.*${dataset}.*_level_7_.*${solution_string}.*"))
#
##all_level_1_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_5_.*${solution_string}.*" |cut -d_ -f5 ))
#all_level_1_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_1.pth" |cut -d_ -f6 ))
#all_level_4_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_4.pth" |cut -d_ -f6 ))
#all_level_7_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_7_.*${solution_string}.*" |cut -d_ -f5 ))
#
#declare -a list_to_use_files=("${level_4_files[@]}")
#declare -a list_to_use_seeds=("${all_level_4_seeds[@]}")
#
#files_level=(0)
#file_seed=(3 4 5 3 4 5 3 4 5 3 4 5 3 4 5)
#
#max=${#files_level[@]}                                  # Take the length of that array
##echo "Beginning of the loop"
##echo "${max}"
#for ((idxA=0; idxA<max; idxA++)); do              # iterate idxA from 0 to length
##  qsub -N "features_${files_level[$idxA]}_seed_${file_seed[$idxA]}" run.sh  "resnet50" "/nobackup/sclaam/checkpoints/resnet50_normal_cifar10_seed_${file_seed[$indxA]}_rf_level_${files_level[$indxA]}_initial_weights.pth" "_no_train_seed_${file_seed[$idxA]}_rf_level_${files_level[$idxA]}"  "${files_level[$idxA]}" "alternative"
##  qsub -l coproc_p100=1 -l h_rt=01:00:00 -N "features_7_seed_${list_to_use_seeds[$idxA]}" run.sh  "resnet50" "${directory}/${list_to_use_files[$idxA]}" "_trained_seed_${list_to_use_seeds[$idxA]}_rf_level_7"  "7" "alternative"
#qsub -l coproc_p100=1 -l h_rt=4:00:00 -N "logistic_features_train_4_seed_${list_to_use_seeds[$idxA]}" run.sh  "resnet50" "${directory}/${list_to_use_files[$idxA]}" "_trained_seed_${list_to_use_seeds[$idxA]}_rf_level_4" "4" "alternative" "1"
#qsub -l coproc_p100=1 -l h_rt=4:00:00 -N "logistic_features_test_4_seed_${list_to_use_seeds[$idxA]}"  run.sh "resnet50" "${directory}/${list_to_use_files[$idxA]}" "_trained_seed_${list_to_use_seeds[$idxA]}_rf_level_4" "4" "alternative" "0"
##  qsub -l coproc_p100=1 -l h_rt=15:00:00 -N "Similarity${list_to_use_seeds[$idxA]}" run.sh  "resnet50" "${directory}/${list_to_use_files[$idxA]}" "trained_seed_${list_to_use_seeds[$idxA]}_rf_level_7"  "7" "alternative"
##echo "solution: ${list_to_use_files[$idxA]} , seed in the same index: ${list_to_use_seeds[$idxA]}"
#  if [ $idxA -gt 0 ]
#  then
#  break
#  fi
#done
#declare -a list_to_use_files=("${level_7_files[@]}")
#declare -a list_to_use_seeds=("${all_level_7_seeds[@]}")
#
#for ((idxA=0; idxA<max; idxA++)); do              # iterate idxA from 0 to length

#qsub -l coproc_p100=1 -l h_rt=01:00:00 -N "test_training_logistic"  run.sh "resnet50" "solution_not_used" "_trained_seed_${list_to_use_seeds[$idxA]}_rf_level_4" "4" "alternative" "47"


#qsub -l coproc_p100=1 -l h_rt=4:00:00 -N "logistic_features_train_7_seed_${list_to_use_seeds[$idxA]}" run.sh  "resnet50" "${directory}/${list_to_use_files[$idxA]}" "_trained_seed_${list_to_use_seeds[$idxA]}_rf_level_7" "7" "alternative" "1"
#qsub -l coproc_p100=1 -l h_rt=4:00:00 -N "logistic_features_test_7_seed_${list_to_use_seeds[$idxA]}"  run.sh "resnet50" "${directory}/${list_to_use_files[$idxA]}" "_trained_seed_${list_to_use_seeds[$idxA]}_rf_level_7" "7" "alternative" "0"
#
#
#  if [ $idxA -gt 0 ]
#  then
#  break
#  fi
#
#done

#declare -a list_to_use_files=("${level_7_files[@]}")
#declare -a list_to_use_seeds=("${all_level_7_seeds[@]}")
#for ((idxA=0; idxA<max; idxA++)); do              # iterate idxA from 0 to length

#qsub -l coproc_p100=1 -l h_rt=01:00:00 -N "test_training_logistic"  run.sh "resnet50" "solution_not_used" "_trained_seed_${list_to_use_seeds[$idxA]}_rf_level_4" "4" "alternative" "47"

#
#qsub -l coproc_p100=1 -l h_rt=4:00:00 -N "logistic_features_train_1_seed_${list_to_use_seeds[$idxA]}" run.sh  "resnet50" "${directory}/${list_to_use_files[$idxA]}" "_trained_seed_${list_to_use_seeds[$idxA]}_rf_level_1" "1" "alternative" "1"
#qsub -l coproc_p100=1 -l h_rt=4:00:00 -N "logistic_features_test_1_seed_${list_to_use_seeds[$idxA]}"  run.sh "resnet50" "${directory}/${list_to_use_files[$idxA]}" "_trained_seed_${list_to_use_seeds[$idxA]}_rf_level_1" "1" "alternative" "0"
#
#
#  if [ $idxA -gt 0 ]
#  then
#  break
#  fi
#
#done

##############################################################################################

#levels_max=${#rf_levels[@]}                                  # Take the length of that array
#seeds_max=${#seeds[@]}                                  # Take the length of that array
#for ((idxA=0; idxA<levels_max; idxA++)); do              # iterate idxA from 0 to length
#for ((idxB=0; idxB<seeds_max; idxB++)); do              # iterate idxA from 0 to length
#for ((idxC=idxB+1; idxC<seeds_max; idxC++)); do              # iterate idxA from 0 to length
#
##echo "seed_${seeds[$idxB]}_VS_seed_${seeds[$idxC]}_level_${rf_levels[$idxA]}"
#qsub -N "similarity_level_${rf_levels[$idxA]}_seeds_${seeds[$idxB]}_${seeds[$idxC]}" run.sh  "resnet50" "_no_train_seed_${seeds[$idxB]}_rf_level_${rf_levels[$idxA]}" "_no_train_seed_${seeds[$idxC]}_rf_level_${rf_levels[$idxA]}" "alternative" "alternative" "npy" "npy"

#done
#done
#done
#########################################################
#             Transform seed name into normal number
#########################################################

########################################################################################################################
#                 Hessian spectra of solutions
########################################################################################################################
#
#model="resnet50"
#dataset="cifar10"
#init=true
#if [ $init ]; then
#    solution_string="initial_weights"
#
#else
#
#
#    solution_string="test_acc"
#fi
#
#
#directory=/nobackup/sclaam/checkpoints
#resnet50_normal_cifar10_seed_0_3_rf_level_1_initial_weights.pth
#level_1_seed0=($(ls $directory | grep -i "${model}.*${dataset}.*_seed_0_.*_level_1_${solution_string}.*"))
#echo $level_1_seed0
##level_1_seed1=($(ls $directory | grep -i "vgg19.*_seed_1_rf_level_1_test_acc.*"))
##echo $level_1_seed1
##
##level1_seeds=($level_1_seed0 $level_1_seed1)
##
##level_2_seed0=($(ls $directory | grep -i "vgg19.*_seed_0_rf_level_2_test_acc.*"))
#level_2_seed0=($(ls $directory | grep -i "${model}.*${dataset}.*_seed_0_.*_level_2_${solution_string}.*"))
#echo $level_2_seed0
##level_2_seed1=($(ls $directory | grep -i "vgg19.*_seed_1_rf_level_2_test_acc.*"))
##echo $level_2_seed1
##
##level2_seeds=($level_2_seed0 $level_2_seed1)
##
#level_3_seed0=($(ls $directory | grep -i "${model}.*${dataset}.*_seed_0_.*_level_3_${solution_string}.*"))
##level_3_seed0=($(ls $directory | grep -i "vgg19.*_seed_0_rf_level_3_test_acc.*"))
##
#echo $level_3_seed0
##level_3_seed1=($(ls $directory | grep -i "vgg19.*_seed_1_rf_level_3_test_acc.*"))
##echo $level_3_seed1
##
##level3_seeds=($level_3_seed0 $level_3_seed1)
##
#
##level_4_seed0=($(ls $directory | grep -i "vgg19.*_seed_0_rf_level_4_test_acc.*"))
#level_4_seed0=($(ls $directory | grep -i "${model}.*${dataset}.*_seed_0_.*_rf_level_4_${solution_string}.*"))
#echo $level_4_seed0
##level_4_seed1=($(ls $directory | grep -i "vgg19.*_seed_1_rf_level_4_test_acc.*"))
##echo $level_4_seed1
##level4_seeds=($level_4_seed0 $level_4_seed1)
#
#
#seeds="0"
#rf_levels=(1 2 3 4)
#levels_max=${#rf_levels[@]}                                  # Take the length of that array
#for ((idxA=0; idxA<levels_max; idxA++)); do              # iterate idxA from 0 to length
#
#
##levels_by_seed=(${level1_seeds[$idxB]} ${level2_seeds[$idxB]} ${level3_seeds[$idxB]} ${level4_seeds[$idxB]})
#
#levels_by_seed=(${level_1_seed0} ${level_2_seed0} ${level_3_seed0} ${level_4_seed0})
#echo "${directory}/${levels_by_seed[$idxA]}"
##echo "${levels_by_seed}"
#
#qsub -N "${model}_hessian_init_${dataset}_${rf_levels[$idxA]}" run.sh  "${model}" "${dataset}" "${rf_levels[$idxA]}" "normal" "seed_0_rf_level_${rf_levels[$idxA]}_init" "${directory}/${levels_by_seed[$idxA]}"
#
#done
#
########################################################################################################################
#                 Creating features loop and other not comparative experiments
########################################################################################################################

#seeds=(0 1)
#rf_levels=(1 2 3 4)
#levels_max=${#rf_levels[@]}                                  # Take the length of that array
#seeds_max=${#seeds[@]}                                  # Take the length of that array
#for ((idxA=0; idxA<levels_max; idxA++)); do              # iterate idxA from 0 to length
#for ((idxB=0; idxB<seeds_max; idxB++)); do              # iterate idxA from 0 to length
#
#
#levels_by_seed=(${level1_seeds[$idxB]} ${level2_seeds[$idxB]} ${level3_seeds[$idxB]} ${level4_seeds[$idxB]})
#
#
#
##temp=${levels_by_seed[$idxA]}
##echo $temp
##echo ${level1_seeds[$idxB]}
#echo "level ${rf_levels[$idxA]} seed ${seeds[$idxB]}"
#echo "solution ${levels_by_seed[$idxA]}"
#
#qsub -N "features_vgg19_${rf_levels[$idxA]}_seed_${seeds[$idxB]}" run.sh  "vgg19" "${directory}/${levels_by_seed[$idxA]}" "_seed_${seeds[$idxB]}_rf_level_${rf_levels[$idxA]}"  "${rf_levels[$idxA]}" "alternative"
#
##levels_by_seed=(${level1_seeds[0]} ${level2_seeds[0]} ${level3_seeds[0]} ${level4_seeds[0]})
#
##qsub -N "vgg19_pruning_summary_level_${rf_levels[$idxA]}" run.sh "vgg19" "cifar10" "2" "${rf_levels[$idxA]}" "normal" "${directory}"
##qsub -N "resnet50_pruning_summary_level_${rf_levels[$idxA]}" run.sh "resnet50" "cifar10" "2" "${rf_levels[$idxA]}" "normal" "${directory}"
#
#
##done
#done
#

########################################################################################################################
#                 Prune summary
########################################################################################################################





#
#directory=/nobackup/sclaam/checkpoints
#model="vgg19"
#dataset="small_imagenet"
###seeds=(0 1 2 3 4)
#rf_levels=(5 6 7)
#levels_max=${#rf_levels[@]}                                  # Take the length of that array
##number_of_elements_by_seed=${#[@]}
#for ((idxA=0; idxA<levels_max; idxA++)); do              # iterate idxA from 0 to length
#qsub -l coproc_p100=1 -N "${model}_${dataset}_pruning_summary_level_${rf_levels[$idxA]}" run.sh "${model}" "${dataset}" "2" "${rf_levels[$idxA]}" "normal" "${directory}"
#done

########################################################################################################################
#                 Prune and fine tune summary
########################################################################################################################
#
#
#model="vgg19"
#dataset="cifar10"
#init=0
#experiment=1
##solution_string="initial_weights"
#solution_string="test_acc"
#directory=/nobackup/sclaam/checkpoints
#level_1_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_1_${solution_string}.*"))
##level_1_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_1.pth"))
#
#level_2_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_2_${solution_string}.*"))
##level_2_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_2.pth"))
#level_3_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_3_${solution_string}.*"))
##level_3_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_3.pth"))
#level_4_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_4_${solution_string}.*"))
##level_4_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_4.pth"))
#declare -a list_to_use=("${level_1_seeds[@]}")


#                                               AA REALLLLLL    Pruning summaries one shot





#model="resnet_small"

model="resnet18"
#model="resnet50"
#model="vgg19"

dataset="small_imagenet"
#dataset="cifar10"
#dataset="cifar100"
directory=/nobackup/sclaam/checkpoints

#directory=/home/luisaam/Documents/PhD/checkpoints


##seeds=(0 1 2)
# Pruning rates for Sotchastic Pruning
pruning_rates=("0.6" "0.7" "0.8" "0.9" "0.95")
# Pruning rates for Receptive field
#pruning_rates=("0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9")
#pruning_rates=("0.9")
sigma=("0.001" "0.003" "0.005")
# For resnet18
#
rf_levels=("0.001" "0.003" "0.005")

#rf_levels=("0.005")

#rf_levels=('k6' 'k7' 'k8')

######     RF for vgg

#rf_levels=(3 4 5)

#rf_levels=(3 4 5 "k6" "k7")


#rf_levels=("k6" "k7" "k8")


#rf_levels=(3 5 7)
levels_max=${#rf_levels[@]}                                  # Take the length of that array
#seeds_per_level=${#list_to_use[@]}                            # Take the length of that array
number_pruning_rates=${#pruning_rates[@]}                            # Take the length of that array

for ((idxA=0; idxA<number_pruning_rates; idxA++)); do                # iterate idxA from 0 to length
for ((idxB=0; idxB<levels_max; idxB++));do              # iterate idxB from 0 to length

#qsub -N "${model}_${dataset}pruning_fine_tuning_summary_level_1_${pruning_rates[$idxB]}" run.sh "${model}" "${dataset}" "2" "1" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}" "0.9" "2"

#qsub -l coproc_v100=1 -l h_rt=01:00:00 -N "${model}_${dataset}_pruning_summary_${rf_levels[$idxB]}_${pruning_rates[$idxA]}" run.sh "${model}" "${dataset}" "4" "${rf_levels[$idxB]}" "normal" "${directory}" "${pruning_rates[$idxA]}" "1"

qsub -l coproc_v100=1 -l h_rt=02:00:00 -N "${model}_${dataset}_soup_idea_${rf_levels[$idxB]}_${pruning_rates[$idxA]}" run.sh "${pruning_rates[$idxA]}" "${model}" "${rf_levels[$idxB]}" "${dataset}"

#python main.py --experiment 1 --batch_size 518 --modeltype "alternative" --pruner "global" --population 5 --epochs 10 --pruning_rate --architecture "${model}" --sigma "${rf_levels[$idxB]}" --dataset "${dataset}"

#which python
#./run.sh "${model}" "${dataset}" "4" "${rf_levels[$idxB]}" "normal" "${directory}" "${pruning_rates[$idxA]}" "1"
##./run.sh "${model}" "${dataset}" "2" "1" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}" "0.5" "1"

done
done




#declare -a list_to_use=("${level_2_seeds[@]}")
#
#seeds_per_level=${#list_to_use[@]}                            # Take the length of that array
##for ((idxA=0; idxA<levels_max; idxA++)); do                # iterate idxA from 0 to length
##for ((idxA=0; idxA<number_pruning_rates; idxA++)); do                # iterate idxA from 0 to length
#for ((idxB=0; idxB<seeds_per_level; idxB++));do              # iterate idxB from 0 to length
#qsub -N "${model}_${dataset}pruning_fine_tuning_summary_level_2_${pruning_rates[$idxB]}" run.sh "${model}" "${dataset}" "2" "2" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}" "0.9" "2"
##echo "${model}" "${dataset}" "2" "1" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}"
##done
#done
#
#
#declare -a list_to_use=("${level_3_seeds[@]}")
#
#seeds_per_level=${#list_to_use[@]}                            # Take the length of that array
##for ((idxA=0; idxA<levels_max; idxA++)); do                # iterate idxA from 0 to length
##for ((idxA=0; idxA<number_pruning_rates; idxA++)); do                # iterate idxA from 0 to length
#for ((idxB=0; idxB<seeds_per_level; idxB++));do              # iterate idxB from 0 to length
#qsub -N "${model}_${dataset}pruning_fine_tuning_summary_level_3_${pruning_rates[$idxB]}" run.sh "${model}" "${dataset}" "2" "3" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}" "0.9" "2"
##echo "${model}" "${dataset}" "2" "1" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}"
##done
#done
#
#declare -a list_to_use=("${level_4_seeds[@]}")
#
#seeds_per_level=${#list_to_use[@]}                            # Take the length of that array
##for ((idxA=0; idxA<levels_max; idxA++)); do                # iterate idxA from 0 to length
##for ((idxA=0; idxA<number_pruning_rates; idxA++)); do                # iterate idxA from 0 to length
#for ((idxB=0; idxB<seeds_per_level; idxB++));do              # iterate idxB from 0 to length
#qsub -N "${model}_${dataset}pruning_fine_tuning_summary_level_4_${pruning_rates[$idxB]}" run.sh "${model}" "${dataset}" "2" "4" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}" "0.9" "2"
##echo "${model}" "${dataset}" "2" "1" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}"
#done
##done
#
#
#########################################################################################################################
#model="vgg19"
#dataset="tiny_imagenet"
#init=0
##solution_string="initial_weights"
#solution_string="test_acc"
#directory=/nobackup/sclaam/checkpoints
#level_1_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_1_${solution_string}.*"))
##level_1_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_1.pth"))
#
#level_2_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_2_${solution_string}.*"))
##level_2_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_2.pth"))
#level_3_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_3_${solution_string}.*"))
##level_3_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_3.pth"))
#level_4_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_4_${solution_string}.*"))
##level_4_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_4.pth"))
#declare -a list_to_use=("${level_1_seeds[@]}")
#
##model="resnet50"
##dataset="tiny_imagenet"
##directory=/nobackup/sclaam/checkpoints
#
##seeds=(0 1 2)
##rf_levels=(1 4)
##levels_max=${#rf_levels[@]}                                  # Take the length of that array
#seeds_per_level=${#list_to_use[@]}                            # Take the length of that array
##for ((idxA=0; idxA<levels_max; idxA++)); do                # iterate idxA from 0 to length
##for ((idxA=0; idxA<number_pruning_rates; idxA++)); do                # iterate idxA from 0 to length
#for ((idxB=0; idxB<seeds_per_level; idxB++));do              # iterate idxB from 0 to length
#qsub -N "${model}_${dataset}pruning_fine_tuning_summary_level_4_${pruning_rates[$idxB]}" run.sh "${model}" "${dataset}" "2" "1" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}" "0.9" "2"
##echo "${model}" "${dataset}" "2" "1" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}"
##done
#done
#
#
#declare -a list_to_use=("${level_2_seeds[@]}")
#
#seeds_per_level=${#list_to_use[@]}                            # Take the length of that array
##for ((idxA=0; idxA<levels_max; idxA++)); do                # iterate idxA from 0 to length
#
##for ((idxA=0; idxA<number_pruning_rates; idxA++)); do                # iterate idxA from 0 to length
#for ((idxB=0; idxB<seeds_per_level; idxB++));do              # iterate idxB from 0 to length
#qsub -N "${model}_${dataset}pruning_fine_tuning_summary_level_4_${pruning_rates[$idxB]}" run.sh "${model}" "${dataset}" "2" "2" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}" "0.9" "2"
##echo "${model}" "${dataset}" "2" "1" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}"
##done
#done
#
#
#declare -a list_to_use=("${level_3_seeds[@]}")
#
#seeds_per_level=${#list_to_use[@]}                            # Take the length of that array
##for ((idxA=0; idxA<levels_max; idxA++)); do                # iterate idxA from 0 to length
##for ((idxA=0; idxA<number_pruning_rates; idxA++)); do                # iterate idxA from 0 to length
#for ((idxB=0; idxB<seeds_per_level; idxB++));do              # iterate idxB from 0 to length
#qsub -N "${model}_${dataset}pruning_fine_tuning_summary_level_3_${pruning_rates[$idxB]}" run.sh "${model}" "${dataset}" "2" "3" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}" "0.9" "2"
##echo "${model}" "${dataset}" "2" "1" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}"
##done
#done
#
#declare -a list_to_use=("${level_4_seeds[@]}")
#
#seeds_per_level=${#list_to_use[@]}                            # Take the length of that array
##for ((idxA=0; idxA<levels_max; idxA++)); do                # iterate idxA from 0 to length
##for ((idxA=0; idxA<number_pruning_rates; idxA++)); do                # iterate idxA from 0 to length
#for ((idxB=0; idxB<seeds_per_level; idxB++));do              # iterate idxB from 0 to length
#qsub -N "${model}_${dataset}pruning_fine_tuning_summary_level_4_${pruning_rates[$idxB]}" run.sh "${model}" "${dataset}" "2" "4" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}" "0.9" "2"
##echo "${model}" "${dataset}" "2" "1" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}"
##done
#done
#
#
########################################################################################################################
#         fine tuning solutions

#model="vgg19"
#dataset="tiny_imagenet"
#pruning_rates=("0.5" "0.6" "0.7" "0.8")
#number_pruning_rates=${#pruning_rates[@]}                            # Take the length of that array
#init=0
#solution_string="initial_weights"
#solution_string="test_acc"
#directory=/nobackup/sclaam/checkpoints
#level_1_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_1_${solution_string}.*"))
#level_1_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_1.pth"))

#level_2_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_2_${solution_string}.*"))
#level_2_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_2.pth"))
#level_3_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_3_${solution_string}.*"))
#level_3_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_3.pth"))
#level_4_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_4_${solution_string}.*"))
#level_4_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_4.pth"))

#declare -a list_to_use=("${level_1_seeds[@]}")







#model="vgg19"
#dataset="tiny_imagenet"
#directory=/nobackup/sclaam/checkpoints
#directory=/home/luisaam/PycharmProjects/sparse_ensemble/trained_models
#seeds=(0 1 2)
#rf_levels=(1 2 3 4)
#levels_max=${#rf_levels[@]}                                  # Take the length of that array
#pruning_rates=(0.9)
#sigmas=("0.0005" "0.0003" "0.0001")
#sigmas_max=${#sigmas[@]}                                  # Take the length of that array
#
#seeds_per_level=${#list_to_use[@]}                            # Take the length of that array
#
#for ((idxA=0; idxA<sigmas_max; idxA++)); do                # iterate idxA from 0 to length
#
#for ((idxB=0; idxB<levels_max; idxB++));do              # iterate idxB from 0 to length

#qsub -l coproc_p100=1 -l h_rt=3:00:00 -N "${model}_${dataset}_SP_pruning_summary_level_${rf_levels[$idxB]}_0.9_${sigmas[$idxA]}" run.sh "${model}" "${dataset}" "4" "${rf_levels[$idxB]}" "normal" "${directory}" "0.9" "${sigmas[$idxA]}"

#echo "${model}" "${dataset}" "2" "1" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}"

#qsub -l coproc_p100=1 -l h_rt=15:00:00 -N "${model}_${dataset}_n_shallow_summary_level_${rf_levels[$idxB]}_more_time" run.sh "${model}" "${dataset}" "4" "${rf_levels[$idxB]}" "normal" "${directory}" "4"

#qsub -l coproc_p100=1 -l h_rt=1:00:00 -N "${model}_${dataset}_pruning_summary_level_${rf_levels[$idxB]}" run.sh "${model}" "${dataset}" "2" "${rf_levels[$idxB]}" "normal" "${directory}" "1" "no_recording"


#
#done
#done















#qsub -N "${model}_${dataset}_pruning_summary_level_${rf_levels[$idxB]}" run.sh "${model}" "${dataset}" "4" "${rf_levels[$idxB]}" "normal" "${directory}" #"${[$idxB]}" "3"


#declare -a list_to_use=("${level_2_seeds[@]}")
#
#seeds_per_level=${#list_to_use[@]}                            # Take the length of that array
#
##for ((idxA=0; idxA<levels_max; idxA++)); do                # iterate idxA from 0 to length
#
#for ((idxB=0; idxB<seeds_per_level; idxB++));do              # iterate idxB from 0 to length
#
#for ((idxA=0; idxA<number_pruning_rates; idxA++)); do                # iterate idxA from 0 to length
#
#qsub -N "${model}_${dataset}pruning_summary_level_2" run.sh "${model}" "${dataset}" "2" "2" "normal" "${directory}" "${list_to_use[$idxB]}" "3"
#
##echo "${model}" "${dataset}" "2" "1" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}"
##done
#done

#
#declare -a list_to_use=("${level_3_seeds[@]}")
#
#seeds_per_level=${#list_to_use[@]}                            # Take the length of that array
##for ((idxA=0; idxA<levels_max; idxA++)); do                # iterate idxA from 0 to length
#for ((idxA=0; idxA<number_pruning_rates; idxA++)); do                # iterate idxA from 0 to length
#for ((idxB=0; idxB<seeds_per_level; idxB++));do              # iterate idxB from 0 to length
#qsub -N "${model}_${dataset}pruning_summary_level_3_${pruning_rates[$idxA]}" run.sh "${model}" "${dataset}" "2" "3" "normal" "${directory}" "${pruning_rates[$idxA]}" "1"
#qsub -N "${model}_${dataset}pruning_summary_level_3" run.sh "${model}" "${dataset}" "2" "3" "normal" "${directory}" "${list_to_use[$idxB]}" "3"
##echo "${model}" "${dataset}" "2" "1" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}"
##done
#done

#declare -a list_to_use=("${level_4_seeds[@]}")

#seeds_per_level=${#list_to_use[@]}                            # Take the length of that array
#for ((idxA=0; idxA<levels_max; idxA++)); do                # iterate idxA from 0 to length
#for ((idxB=0; idxB<seeds_per_level; idxB++));do              # iterate idxB from 0 to length
#for ((idxA=0; idxA<number_pruning_rates; idxA++)); do                # iterate idxA from 0 to length

#qsub -N "${model}_${dataset}_fine_Tuning_pruning_level_4_${idxB}" run.sh "${model}" "${dataset}" "0" "4" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}" "0.9" "2"

#qsub -N "${model}_${dataset}pruning_summary_level_4_${idxB}" run.sh "${model}" "${dataset}" "2" "4" "normal" "${directory}" "${list_to_use[$idxB]}" "3"

#echo "${model}" "${dataset}" "2" "1" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}"
#done
#done


#
#declare -a list_to_use=("${level_3_seeds[@]}")
#
#seeds_per_level=${#list_to_use[@]}                            # Take the length of that array
##for ((idxA=0; idxA<levels_max; idxA++)); do                # iterate idxA from 0 to length
#for ((idxB=0; idxB<seeds_per_level; idxB++));do              # iterate idxB from 0 to length
##for ((idxA=0; idxA<number_pruning_rates; idxA++)); do                # iterate idxA from 0 to length
#
#qsub -N "${model}_${dataset}_fine_Tuning_pruning_level_3_${idxB}" run.sh "${model}" "${dataset}" "0" "3" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}" "0.9" "2"
#
##qsub -N "${model}_${dataset}pruning_summary_level_4_${idxB}" run.sh "${model}" "${dataset}" "2" "4" "normal" "${directory}" "${list_to_use[$idxB]}" "3"
#
##echo "${model}" "${dataset}" "2" "1" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}"
#done
#
#
#
#
#
#declare -a list_to_use=("${level_2_seeds[@]}")
#
#seeds_per_level=${#list_to_use[@]}                            # Take the length of that array
##for ((idxA=0; idxA<levels_max; idxA++)); do                # iterate idxA from 0 to length
#for ((idxB=0; idxB<seeds_per_level; idxB++));do              # iterate idxB from 0 to length
##for ((idxA=0; idxA<number_pruning_rates; idxA++)); do                # iterate idxA from 0 to length
#
#qsub -N "${model}_${dataset}_fine_Tuning_pruning_level_2_${idxB}" run.sh "${model}" "${dataset}" "0" "2" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}" "0.9" "2"
#
##qsub -N "${model}_${dataset}pruning_summary_level_4_${idxB}" run.sh "${model}" "${dataset}" "2" "4" "normal" "${directory}" "${list_to_use[$idxB]}" "3"
#
##echo "${model}" "${dataset}" "2" "1" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}"
#done
#
#declare -a list_to_use=("${level_1_seeds[@]}")
#
#seeds_per_level=${#list_to_use[@]}                            # Take the length of that array
##for ((idxA=0; idxA<levels_max; idxA++)); do                # iterate idxA from 0 to length
#for ((idxB=0; idxB<seeds_per_level; idxB++));do              # iterate idxB from 0 to length
##for ((idxA=0; idxA<number_pruning_rates; idxA++)); do                # iterate idxA from 0 to length
#
#qsub -N "${model}_${dataset}_fine_Tuning_pruning_level_1_${idxB}" run.sh "${model}" "${dataset}" "0" "2" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}" "0.9" "2"
#
##qsub -N "${model}_${dataset}pruning_summary_level_4_${idxB}" run.sh "${model}" "${dataset}" "2" "4" "normal" "${directory}" "${list_to_use[$idxB]}" "3"
#
##echo "${model}" "${dataset}" "2" "1" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}"
#done
########################################################################################################################
#
#
#model="resnet50"
#dataset="tiny_imagenet"
#init=0
##solution_string="initial_weights"
#solution_string="test_acc"
#directory=/nobackup/sclaam/checkpoints
##level_1_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_1_${solution_string}.*"))
##level_1_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_1.pth"))
#
#level_2_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_2_${solution_string}.*"))
##level_2_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_2.pth"))
#level_3_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_3_${solution_string}.*"))
##level_3_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_3.pth"))
#level_4_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_4_${solution_string}.*"))
##level_4_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_4.pth"))
##
##declare -a list_to_use=("${level_1_seeds[@]}")
###model="resnet50"
###dataset="tiny_imagenet"
###directory=/nobackup/sclaam/checkpoints
##
###seeds=(0 1 2)
###rf_levels=(1 4)
###levels_max=${#rf_levels[@]}                                  # Take the length of that array
##seeds_per_level=${#list_to_use[@]}                            # Take the length of that array
###for ((idxA=0; idxA<levels_max; idxA++)); do                # iterate idxA from 0 to length
##for ((idxB=0; idxB<seeds_per_level; idxB++));do              # iterate idxB from 0 to length
##qsub -N "${model}_${dataset}pruning_fine_tuning_summary_level_4_${idxB}" run.sh "${model}" "${dataset}" "2" "1" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}"
###echo "${model}" "${dataset}" "2" "1" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}"
##done




#declare -a list_to_use=("${level_2_seeds[@]}")

#seeds_per_level=${#list_to_use[@]}                            # Take the length of that array
##for ((idxA=0; idxA<levels_max; idxA++)); do                # iterate idxA from 0 to length
##for ((idxA=0; idxA<number_pruning_rates; idxA++)); do                # iterate idxA from 0 to length
#for ((idxB=0; idxB<seeds_per_level; idxB++));do              # iterate idxB from 0 to length
#qsub -N "${model}_${dataset}pruning_fine_tuning_summary_level_4_${pruning_rates[$idxB]}" run.sh "${model}" "${dataset}" "2" "2" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}" "0.9" "2"
##echo "${model}" "${dataset}" "2" "1" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}"
##done
#done



#declare -a list_to_use=("${level_3_seeds[@]}")

#seeds_per_level=${#list_to_use[@]}                            # Take the length of that array
##for ((idxA=0; idxA<levels_max; idxA++)); do                # iterate idxA from 0 to length
##for ((idxA=0; idxA<number_pruning_rates; idxA++)); do                # iterate idxA from 0 to length
#for ((idxB=0; idxB<seeds_per_level; idxB++));do              # iterate idxB from 0 to length
#qsub -N "${model}_${dataset}pruning_fine_tuning_summary_level_3_${pruning_rates[$idxB]}" run.sh "${model}" "${dataset}" "2" "3" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}" "0.9" "2"
##echo "${model}" "${dataset}" "2" "1" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}"
##done
#done
#
#declare -a list_to_use=("${level_4_seeds[@]}")
#
#seeds_per_level=${#list_to_use[@]}                            # Take the length of that array
##for ((idxA=0; idxA<levels_max; idxA++)); do                # iterate idxA from 0 to length
##for ((idxA=0; idxA<number_pruning_rates; idxA++)); do                # iterate idxA from 0 to length
#for ((idxB=0; idxB<seeds_per_level; idxB++));do              # iterate idxB from 0 to length
#qsub -N "${model}_${dataset}pruning_fine_tuning_summary_level_4_${pruning_rates[$idxB]}" run.sh "${model}" "${dataset}" "2" "4" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}" "0.9" "2"
##echo "${model}" "${dataset}" "2" "1" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}"
##done
#done
##done













###############################################################################
#                 Similarity between seeds loop
###############################################################################

#
#seeds=(0 1 2)
#rf_levels=(5 6 7)
#levels_max=${#rf_levels[@]}                                  # Take the length of that array
#seeds_max=${#seeds[@]}                                  # Take the length of that array
#for ((idxA=0; idxA<levels_max; idxA++)); do              # iterate idxA from 0 to length
#for ((idxB=0; idxB<seeds_max; idxB++)); do              # iterate idxA from 0 to length
#for ((idxC=idxB+1; idxC<seeds_max; idxC++)); do              # iterate idxA from 0 to length
#
###echo "seed_${seeds[$idxB]}_VS_seed_${seeds[$idxC]}_level_${rf_levels[$idxA]}"
##
#qsub -l coproc_p100=1 -l h_rt=20:00:00 -N "similarity_level_${rf_levels[$idxA]}_seeds_${seeds[$idxB]}_${seeds[$idxC]}" run.sh  "resnet50"  "_trained_seed_${seeds[$idxB]}_rf_level_${rf_levels[$idxA]}" "trained_seed_${seeds[$idxC]}_rf_level_${rf_levels[$idxA]}" "alternative" "alternative" "npy" "npy"
#echo "Begin statistics"
#qsub -l coproc_p100=1 -l h_rt=1:00:00 -N "statistics_features_level_1_seeds_0" run.sh  "resnet50"  "_trained_seed_0_rf_level_7" "_trained_seed_0_rf_level_7" "alternative" "alternative" "npy" "npy"
#qsub -l coproc_p100=1 -l h_rt=1:00:00 -N "statistics_features_level_4_seeds_0" run.sh  "resnet50"  "_trained_seed_0_rf_level_7" "_trained_seed_0_rf_level_7" "alternative" "alternative" "npy" "npy"
#qsub -l coproc_p100=1 -l h_rt=00:30:00 -N "statistics_features_level_7_seeds_0" run.sh  "resnet50"  "_trained_seed_0_rf_level_7" "_trained_seed_0_rf_level_7" "alternative" "alternative" "npy" "npy"
#
##trained_seed_${list_to_use_seeds[$idxA]}_rf_level_7
#done
#done
#done
#


#######################################################################################################################
#                                    Prune models for a particular RF level, architecture and dataset
#######################################################################################################################



#directory=/nobackup/sclaam/checkpoints
#rf_levels=(2 3 4)
#levels_max=${#rf_levels[@]}                                  # Take the length of that array
#for ((idxA=0; idxA<levels_max; idxA++)); do              # iterate idxA from 0 to length
#
#
#
#
#
#echo "solution ${levels_by_seed[$idxA]}"
#
#
#
#qsub -N "resnet50_tiny_imagenet_gradient_flow_${rf_levels[$idxA]}" run.sh "resnet50" "tiny_imagenet" "2" "${rf_levels[$idxA]}" "normal" "${directory}"
##qsub -N "vgg19_tiny_imagenet_gradient_flow_${rf_levels[$idxA]}" run.sh "vgg19" "tiny_imagenet" "2" "${rf_levels[$idxA]}" "normal" "${directory}"
#
#
#
#
#done


###############################################################################
#                  This is for changing names
###############################################################################

#directory=/nobackup/sclaam/checkpoints

# all_level_1_seeds=($(ls $directory | grep -i "resnet50_normal_tiny_imagenet.*_level_1_.*" |cut -d_ -f5 |uniq))
# echo $all_level_1_seeds
# all_level_2_seeds=($(ls $directory | grep -i "resnet50_normal_tiny_imagenet.*_level_2_.*" |cut -d_ -f5 |uniq))
# echo $all_level_2_seeds
# all_level_3_seeds=($(ls $directory | grep -i "resnet24_normal_tiny_imagenet.*_level_3_.*no_recording.*" |cut -d_ -f5 |uniq))
# echo $all_level_3_seeds
# all_level_4_seeds=($(ls $directory | grep -i "resnet50_normal_tiny_imagenet.*_level_4_.*" |cut -d_ -f5 |uniq))
# echo $all_level_4_seeds
# all_level_5_seeds=($(ls $directory | grep -i "resnet24_normal_tiny_imagenet_.*_level_5_.*no_recording.*" |cut -d_ -f5 |uniq))
# echo $all_level_5_seeds
# all_level_6_seeds=($(ls $directory | grep -i "resnet50_normal_cifar10_.*_level_6_.*no_recording.*" |cut -d_ -f4 |uniq))
# echo $all_level_6_seeds


#echo " "
#echo "Level 7 \n"
#
#echo " "
#
#
#declare -a list_to_use=("${all_level_7_seeds[@]}")
##
#max=${#list_to_use[@]}                                  # Take the length of that array
##
#echo $max
##
#for ((idxA=0; idxA<max; idxA++)); do # iterate idxA from 0 to length
#echo "${directory}/.*${list_to_use[$idxA]}\.\*"
#file_names=($(ls $directory | grep -i ".*${list_to_use[$idxA]}.*.pth"))
#echo $file_names
#echo ${#file_names[@]}                                  # Take the length of that array
#echo $idxA
#
#for pathname in  "${file_names[@]}"; do
#replace_string="seed_${idxA}"
#thing="${pathname/"${list_to_use[$idxA]}"/$replace_string}"
#  echo "${thing}"
##  mv -i "${directory}/${pathname}" "${directory}/${thing}"
#done
#done
##
##
##
##
#echo " "
#echo "Level 3 "
#echo " "
##
##
#declare -a list_to_use=("${all_level_3_seeds[@]}")
##
#max=${#list_to_use[@]}                                  # Take the length of that array
##
#echo $max
##
#for ((idxA=0; idxA<max; idxA++)); do # iterate idxA from 0 to length
#echo "${directory}/.*${list_to_use[$idxA]}\.\*"
#file_names=($(ls $directory | grep -i ".*${list_to_use[$idxA]}.*.pth"))
#echo $file_names
#echo ${#file_names[@]}                                  # Take the length of that array
#echo $idxA
#
#for pathname in  "${file_names[@]}"; do
#replace_string="seed_${idxA}"
#thing="${pathname/"${list_to_use[$idxA]}"/$replace_string}"
#  echo "${thing}"
##  mv -i "${directory}/${pathname}" "${directory}/${thing}"
#done
#done


########################################################################################################################

#
#print_seed_rename () {
#
#max=${#($2[@])}                                  # Take the length of that array
#echo $max
#for ((idxA=0; idxA<max; idxA++)); do # iterate idxA from 0 to length
#echo "$1/.*${$2[$idxA]}\.\*"
#file_names=($(ls $2 | grep -i ".*${$2[$idxA]}.*.pth"))
#echo $file_names
#echo ${#file_names[@]}                                  # Take the length of that array
#echo $idxA
#
#for pathname in  "${file_names[@]}"; do
#replace_string="seed_${idxA}"
#thing="${pathname/"${$2[$idxA]}"/$replace_string}"
#  echo "${thing}"
##  mv -i "${directory}/${pathname}" "${directory}/${thing}"
##
##    if [[ -f $pathname ]] && grep -q -F "$string" "$pathname"; then
##        mv -i "$pathname" "${pathname%.*}.xml"
##    fi
#done
#done
#}

#print_seed_rename $directory $all_level_2_seeds






