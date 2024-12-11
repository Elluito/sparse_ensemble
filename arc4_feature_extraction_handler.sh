#!/bin/bash


run_extraction() {
model=$1
dataset=$2
data_folder=$3
#directory=$4
solution=$4
#ffcv=$6
#ffcv_train=$7
#ffcv_val=$8
rf_level=$5
resolution=$6
num_workers=$7
batch_size=$8
save_folder=$9
name="${10}"
downsampling="${11}"
latent_folder="${12}"
adjust_bn="${13}"
pruning_rate="${14}"
#pruning_rates=("0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9")
#pruning_rates=("0.5")
# For resnet18
#
#  if [ "${10}" -gt 0 ]




qsub -l h_rt=40:00:00 -l coproc_v100=1  -N "extract_features_${model}_${dataset}_rf_level_${rf_level}" arc4_feature_extraction_run.sh "${data_folder}" "${model}" "normal" "${solution}" "${dataset}" "${rf_level}" "${resolution}" "${num_workers}" "${batch_size}" "${save_folder}" "${name}" "${latent_folder}" "${downsampling}" "${adjust_bn}" "${pruning_rate}"

}

data_folder="/nobackup/sclaam/data"
logs_folder="./probes_logs/"
latent_folder="/nobackup/sclaam/latent_representations/"




model="resnet25_small"
dataset="small_imagenet"
resolution=224

# "solution": "/home/luisaam/Downloads/resnet_25_small_imagenet/resnet25_small_normal_small_imagenet_1731451265.2231426_rf_level_6_sgd_100_res_224_no_ffcv_test_acc_68.76.pth",
# "solution": "/home/luisaam/Downloads/resnet_25_small_imagenet/resnet25_small_normal_small_imagenet_1731451265.2230341_rf_level_7_sgd_100_res_224_no_ffcv_test_acc_67.62.pth",
# "solution": "/home/luisaam/Downloads/resnet_25_small_imagenet/resnet25_small_normal_small_imagenet_1731451265.223035_rf_level_8_sgd_100_res_224_no_ffcv_test_acc_67.23.pth",
# "solution": "/home/luisaam/Downloads/resnet_25_small_imagenet/resnet25_small_normal_small_imagenet_1731451265.2231302_rf_level_10_sgd_100_res_224_no_ffcv_test_acc_65.03.pth",
# "solution": "/home/luisaam/Downloads/resnet_25_small_imagenet/resnet25_small_normal_small_imagenet_1731002508.677377_rf_level_11_sgd_100_res_224_no_ffcv_test_acc_44.11.pth",
# "solution": "/home/luisaam/Downloads/resnet_25_small_imagenet/resnet25_small_normal_small_imagenet_1731002507.845115_rf_level_12_sgd_100_res_224_no_ffcv_test_acc_34.88.pth",
# "solution": "/home/luisaam/Downloads/resnet_25_small_imagenet/resnet25_small_normal_small_imagenet_1731002508.0559256_rf_level_13_sgd_100_res_224_no_ffcv_test_acc_29.1.pth",
sol1rs25="/nobackup/sclaam/resnet_25_small_imagenet/resnet25_small_normal_small_imagenet_1731451265.2229764_rf_level_5_sgd_100_res_224_no_ffcv_test_acc_70.3.pth"
sol2rs25="/nobackup/sclaam/resnet_25_small_imagenet/resnet25_small_normal_small_imagenet_1731451265.2231426_rf_level_6_sgd_100_res_224_no_ffcv_test_acc_68.76.pth"
sol3rs25="/nobackup/sclaam/resnet_25_small_imagenet/resnet25_small_normal_small_imagenet_1731451265.2230341_rf_level_7_sgd_100_res_224_no_ffcv_test_acc_67.62.pth"
sol4rs25="/nobackup/sclaam/resnet_25_small_imagenet/resnet25_small_normal_small_imagenet_1731451265.223035_rf_level_8_sgd_100_res_224_no_ffcv_test_acc_67.23.pth"
sol5rs25="/nobackup/sclaam/resnet_25_small_imagenet/resnet25_small_normal_small_imagenet_1731451265.2231302_rf_level_10_sgd_100_res_224_no_ffcv_test_acc_65.03.pth"
sol6rs25="/nobackup/sclaam/resnet_25_small_imagenet/resnet25_small_normal_small_imagenet_1731002508.677377_rf_level_11_sgd_100_res_224_no_ffcv_test_acc_44.11.pth"
sol7rs25="/nobackup/sclaam/resnet_25_small_imagenet/resnet25_small_normal_small_imagenet_1731002507.845115_rf_level_12_sgd_100_res_224_no_ffcv_test_acc_34.88.pth"
sol8rs25="/nobackup/sclaam/resnet_25_small_imagenet/resnet25_small_normal_small_imagenet_1731002508.0559256_rf_level_13_sgd_100_res_224_no_ffcv_test_acc_29.1.pth"
#resnet25_small_normal_small_imagenet_1731451265.2229764_rf_level_5_sgd_100_res_224_no_ffcv_test_acc_70.3.pth



rf_levels=("5" "6" "7" "8" "10" "11" "12" "13")

solutions=("${sol1rs25}" "${sol2rs25}" "${sol3rs25}" "${sol4rs25}" "${sol5rs25}" "${sol6rs25}" "${sol7rs25}" "${sol8rs25}")
names=("sgd_100_res_224_no_ffcv_test" "sgd_100_res_224_no_ffcv_test" "sgd_100_res_224_no_ffcv_test" "sgd_100_res_224_no_ffcv_test" "sgd_100_res_224_no_ffcv_test" "sgd_100_res_224_no_ffcv_test" "sgd_100_res_224_no_ffcv_test" "sgd_100_res_224_no_ffcv_test")

#names=("seed_0" "seed_0" "seed_0" "seed_0" "seed_0" "seed_0" "seed_0")
levels_max=${#rf_levels[@]}                                  # Take the length of that array
for ((idxA=0; idxA<levels_max; idxA++));do              # iterate idxB from 0 to length
 run_extraction "${model}" "${dataset}" "${data_folder}" "${solutions[$idxA]}" "${rf_levels[$idxA]}" "${resolution}" 16 128 "${logs_folder}" "${names[$idxA]}" 4 "${latent_folder}" 0 "0.8"  # "${names[$idxA]}"  "${save_folder}" "50" "0.001" "0"
done


