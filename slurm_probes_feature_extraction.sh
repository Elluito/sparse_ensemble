#!/user/bin/bash


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
#pruning_rates=("0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9")
#pruning_rates=("0.5")
# For resnet18
#
#  if [ "${10}" -gt 0 ]




sbatch --nodes=1 --time=140:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --gres=gpu:1 --error="extract_features_${model}_${dataset}_rf_level_${rf_level}.err"  --output="extract_features_${model}_${dataset}_rf_level_${rf_level}.out"  --job-name="extract_features_${model}_${dataset}_rf_level_${rf_level}" slurm_probes_extraction_run.sh "${data_folder}" "${model}" "normal" "${solution}" "${dataset}" "${rf_level}" "${resolution}" "${num_workers}" "${batch_size}" "${save_folder}"
}

data_folder="/jmain02/home/J2AD014/mtc03/lla98-mtc03/datasets"
logs_folder="./probes_logs/"



# CIFAR 10

## vgg

save_folder="${HOME}/truncated_models_results"

sol1vgg=/jmain02/home/J2AD014/mtc03/lla98-mtc03/original_paper_checkpoints/vgg19_normal_cifar10_seed.0_rf_level_1_recording_200_no_ffcv_test_acc_93.77.pth
sol2vgg=/jmain02/home/J2AD014/mtc03/lla98-mtc03/original_paper_checkpoints/vgg19_normal_cifar10_seed.0_rf_level_2_recording_200_no_ffcv_test_acc_90.98.pth
sol3vgg=/jmain02/home/J2AD014/mtc03/lla98-mtc03/original_paper_checkpoints/vgg19_normal_cifar10_seed.0_rf_level_3_recording_200_no_ffcv_test_acc_88.13.pth
sol4vgg=/jmain02/home/J2AD014/mtc03/lla98-mtc03/original_paper_checkpoints/vgg19_normal_cifar10_seed.0_rf_level_4_recording_200_no_ffcv_test_acc_86.12.pth

# Resnet50 CIFAR10
sol1rs50=/jmain02/home/J2AD014/mtc03/lla98-mtc03/original_paper_checkpoints/resnet50_normal_cifar10_seed.0_rf_level_1_recording_200_no_ffcv_test_acc_94.99.pth
sol2rs50=/jmain02/home/J2AD014/mtc03/lla98-mtc03/original_paper_checkpoints/resnet50_normal_cifar10_seed.0_rf_level_2_recording_200_no_ffcv_test_acc_94.24.pth
sol3rs50=/jmain02/home/J2AD014/mtc03/lla98-mtc03/original_paper_checkpoints/resnet50_normal_cifar10_seed.0_rf_level_3_recording_200_no_ffcv_test_acc_92.3.pth
sol4rs50=/jmain02/home/J2AD014/mtc03/lla98-mtc03/original_paper_checkpoints/resnet50_normal_cifar10_seed.0_rf_level_4_recording_200_no_ffcv_test_acc_90.91.pth
sol5rs50=/jmain02/home/J2AD014/mtc03/lla98-mtc03/original_paper_checkpoints/resnet50_normal_cifar10_seed.0_rf_level_9_recording_200_no_ffcv_test_acc_73.45.pth
sol6rs50=/jmain02/home/J2AD014/mtc03/lla98-mtc03/original_paper_checkpoints/resnet50_normal_cifar10_seed.0_rf_level_10_recording_200_no_ffcv_test_acc_53.77.pth
sol7rs50=/jmain02/home/J2AD014/mtc03/lla98-mtc03/original_paper_checkpoints/resnet50_normal_cifar10_seed.0_rf_level_11_recording_200_no_ffcv_test_acc_50.06.pth

rf_levels=("1" "2" "3" "4")
solutions=("${sol1vgg}" "${sol2vgg}" "${sol3vgg}" "${sol4vgg}")
names=("seed_0" "seed_0" "seed_0" "seed_0")
levels_max=${#rf_levels[@]}                                  # Take the length of that array
for ((idxA=0; idxA<levels_max; idxA++));do              # iterate idxB from 0 to length
 run_extraction "vgg19" "cifar10" "${data_folder}" "${solutions[$idxA]}" "${rf_levels[$idxA]}" "32" 2 128 "${logs_folder}"  # "${names[$idxA]}"  "${save_folder}" "50" "0.001" "0"
done


## RESNET50


rf_levels=("1" "2" "3" "4" "9" "10" "11")
solutions=("${sol1rs50}" "${sol2rs50}" "${sol3rs50}" "${sol4rs50}" "${sol5rs50}" "${sol6rs50}" "${sol7rs50}")
names=("seed_0" "seed_0" "seed_0" "seed_0" "seed_0" "seed_0" "seed_0")
#names=("seed_0" "seed_0" "seed_0" "seed_0" "seed_0" "seed_0" "seed_0")
levels_max=${#rf_levels[@]}                                  # Take the length of that array
for ((idxA=0; idxA<levels_max; idxA++));do              # iterate idxB from 0 to length
 run_extraction "resnet50" "cifar10" "${data_folder}" "${solutions[$idxA]}" "${rf_levels[$idxA]}" "32" 2 128 "${logs_folder}"  # "${names[$idxA]}"  "${save_folder}" "50" "0.001" "0"
done




# Tiny ImageNet ###################################################################

## vgg

sol1vgg=/jmain02/home/J2AD014/mtc03/lla98-mtc03/original_paper_checkpoints/vgg19_normal_tiny_imagenet_seed.0_rf_level_1_recording_200_no_ffcv_test_acc_60.95.pth
sol2vgg=/jmain02/home/J2AD014/mtc03/lla98-mtc03/original_paper_checkpoints/vgg19_normal_tiny_imagenet_seed.0_rf_level_2_recording_200_no_ffcv_test_acc_53.21.pth
sol3vgg=/jmain02/home/J2AD014/mtc03/lla98-mtc03/original_paper_checkpoints/vgg19_normal_tiny_imagenet_seed.0_rf_level_3_recording_200_no_ffcv_test_acc_43.97.pth
sol4vgg=/jmain02/home/J2AD014/mtc03/lla98-mtc03/original_paper_checkpoints/vgg19_normal_tiny_imagenet_seed.0_rf_level_4_recording_200_no_ffcv_test_acc_39.34.pth


rf_levels=("1" "2" "3" "4")
solutions=("${sol1vgg}" "${sol2vgg}" "${sol3vgg}" "${sol4vgg}")
names=("seed_0" "seed_0" "seed_0" "seed_0")
#levels_max=${#rf_levels[@]}                                  # Take the length of that array
#for ((idxA=0; idxA<levels_max; idxA++));do              # iterate idxB from 0 to length
#
#run_paper_truncated "vgg19" "tiny_imagenet" "${rf_levels[$idxA]}" "${names[$idxA]}" "${solutions[$idxA]}" "${save_folder}" "50" "0.001" "0"
#
#done
#

#idxA=2
#run_paper_truncated "vgg19" "tiny_imagenet" "${rf_levels[$idxA]}" "${names[$idxA]}" "${solutions[$idxA]}" "${save_folder}" "50" "0.001" "0"
#idxA=3
#run_paper_truncated "vgg19" "tiny_imagenet" "${rf_levels[$idxA]}" "${names[$idxA]}" "${solutions[$idxA]}" "${save_folder}" "50" "0.001" "0"

## RESNET50

sol2rs50=/jmain02/home/J2AD014/mtc03/lla98-mtc03/original_paper_checkpoints/resnet50_normal_tiny_imagenet_seed.0_rf_level_2_recording_200_no_ffcv_test_acc_62.78.pth
sol3rs50=/jmain02/home/J2AD014/mtc03/lla98-mtc03/original_paper_checkpoints/resnet50_normal_tiny_imagenet_seed.0_rf_level_3_recording_200_no_ffcv_test_acc_59.46.pth
sol4rs50=/jmain02/home/J2AD014/mtc03/lla98-mtc03/original_paper_checkpoints/resnet50_normal_tiny_imagenet_seed.0_rf_level_4_recording_200_no_ffcv_test_acc_55.59.pth
sol5rs50=/jmain02/home/J2AD014/mtc03/lla98-mtc03/original_paper_checkpoints/resnet50_normal_tiny_imagenet_seed.0_rf_level_9_recording_200_no_ffcv_test_acc_36.0.pth
sol6rs50=/jmain02/home/J2AD014/mtc03/lla98-mtc03/original_paper_checkpoints/resnet50_normal_tiny_imagenet_seed.0_rf_level_10_recording_200_no_ffcv_test_acc_33.18.pth
sol7rs50=/jmain02/home/J2AD014/mtc03/lla98-mtc03/original_paper_checkpoints/resnet50_normal_tiny_imagenet_seed.0_rf_level_11_recording_200_no_ffcv_test_acc_25.44.pth
rf_levels=("2" "3" "4" "9" "10" "11")
solutions=("${sol2rs50}" "${sol3rs50}" "${sol4rs50}" "${sol5rs50}" "${sol6rs50}" "${sol7rs50}")

