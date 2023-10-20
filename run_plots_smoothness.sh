#!/bin/bash -l


# Level 0
rf_level0_s1="smoothness_partial/resnet50/loss_data_fin__seed_1_rf_level_0.pkl"
name_rf_level0_s1="_seed_1_rf_level_0"


rf_level0_s2="smoothness_partial/resnet50/loss_data_fin__seed_2_rf_level_0.pkl"
name_rf_level0_s2="_seed_2_rf_level_0"

# level 1
rf_level1_s1="smoothness_partial/resnet50/loss_data_fin__seed_1_rf_level_1.pkl"
name_rf_level1_s1="_seed_1_rf_level_1"
rf_level1_s2="smoothness_partial/resnet50/loss_data_fin__seed_2_rf_level_1.pkl"
name_rf_level1_s2="_seed_2_rf_level_1"

# level 2
rf_level2_s1="smoothness_partial/resnet50/loss_data_fin__seed_1_rf_level_2.pkl"
name_rf_level2_s1="_seed_1_rf_level_2"

rf_level2_s2="smoothness_partial/resnet50/loss_data_fin__seed_2_rf_level_2.pkl"
name_rf_level2_s2="_seed_2_rf_level_2"
#Level 3

rf_level3_s1="smoothness_partial/resnet50/loss_data_fin__seed_1_rf_level_3.pkl"
name_rf_level3_s1="_seed_1_rf_level_3"

rf_level3_s2="smoothness_partial/resnet50/loss_data_fin__seed_2_rf_level_3.pkl"
name_rf_level3_s2="_seed_2_rf_level_3"


#Level 4
rf_level4_s1="smoothness_partial/resnet50/loss_data_fin__seed_1_rf_level_4.pkl"
name_rf_level4_s1="_seed_1_rf_level_4"
rf_level4_s2="smoothness_partial/resnet50/loss_data_fin__seed_2_rf_level_4.pkl"
name_rf_level4_s2="_seed_2_rf_level_4"

# Pytorch implementation

rf_level_p_s1="smoothness_partial/resnet50/loss_data_fin__seed_1_rf_level_p.pkl"
name_rf_level_p_s1="_seed_1_rf_level_p"

rf_level_p_s2="smoothness_partial/resnet50/loss_data_fin__seed_2_rf_level_p.pkl"
name_rf_level_p_s2="_seed_2_rf_level_p"

rf_level_p_s3="trained_models/cifar10/resnet50_pytorch_cifar10_seed_3_test_acc_89.33.pth"
name_rf_level_p_s3="_seed_3_rf_level_p"






python smoothness_plotting.py --file $rf_level_p_s1 --name $name_rf_level_p_s1 --title "Seed 1 pytorch resnet50 cifar10 "
python smoothness_plotting.py --file $rf_level_p_s2 --name $name_rf_level_p_s2 --title "Seed 2 pytorch resnet50 cifar10 "
python smoothness_plotting.py --file $rf_level0_s1 --name $name_rf_level0_s1 --title "Seed 1 Custom resnet50 cifar10 "
python smoothness_plotting.py --file $rf_level0_s2 --name $name_rf_level0_s2 --title "Seed 1 Custom resnet50 cifar10 "



python smoothness_plotting.py --file $rf_level1_s1 --name $name_rf_level1_s1 --title "Seed 1 Level 1 resnet50 cifar10 "
python smoothness_plotting.py --file $rf_level1_s2 --name $name_rf_level1_s2 --title "Seed 2 Level 1  resnet50 cifar10 "
python smoothness_plotting.py --file $rf_level2_s1 --name $name_rf_level2_s1 --title "Seed 2 Level 2 resnet50 cifar10 "
python smoothness_plotting.py --file $rf_level2_s2 --name $name_rf_level2_s2 --title "Seed 2 Level 2 resnet50 cifar10 "
python smoothness_plotting.py --file $rf_level3_s1 --name $name_rf_level3_s1 --title "Seed 1 Level 3 resnet50 cifar10 "
python smoothness_plotting.py --file $rf_level3_s2 --name $name_rf_level3_s2 --title "Seed 1 Level 3 resnet50 cifar10 "
python smoothness_plotting.py --file $rf_level4_s1 --name $name_rf_level4_s1 --title "Seed 1 Level 4 resnet50 cifar10 "
python smoothness_plotting.py --file $rf_level4_s2 --name $name_rf_level4_s2 --title "Seed 2 Level 4 resnet50 cifar10 "

