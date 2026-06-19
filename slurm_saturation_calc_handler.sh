#!/bin/bash

run_saturation_calc() {

model=$1
dataset=$2
directory=$3
data_folder=$4
save_folder=$5
name=$6
ffcv=$7
ffcv_train=$8
ffcv_val=$9

pruning_rate="${10}"
rf_level="${11}"

if [ "${ffcv}" -gt 0 ]
  then

  echo "Use FFCV"

  sbatch --nodes=1 --time=03:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk  --error="${model}_${rf_level}_${dataset}_${pruning_rate}_${name}_saturation_ffcv.err" --output="${model}_${rf_level}_${dataset}_${pruning_rate}_${name}_ffcv.out" --job-name="${model}_${rf_level}_${dataset}_${pruning_rate}_${name}_no_pr_ffcv" slurm_saturation_calc_run.sh FFCV="${ffcv}" NAME="${name}" MODEL="${model}" DATASET="${dataset}"  NUMW=0  RFL="${rf_level}" TYPE="normal" FOLDER="${directory}" PR="${pruning_rate}" FFCV_TRAIN="${ffcv_train}" FFCV_VAL="${ffcv_val}" DATA_FOLDER="${data_folder}" SAVE_FOLDER="${save_folder}"

else

 echo "Don't use FFCV"

 sbatch --nodes=1 --time=03:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk  --error="${model}_${rf_level}_${dataset}_${pruning_rate}_${name}_saturation_no_ffcv.err" --output="${model}_${rf_level}_${dataset}_${pruning_rate}_${name}_saturation_no_ffcv.out" --job-name="${model}_${rf_level}_${dataset}_${pruning_rate}_${name}_saturation_no_ffcv" slurm_saturation_calc_run.sh FFCV="${ffcv}" NAME="${name}" MODEL="${model}" DATASET="${dataset}"  NUMW=0  RFL="${rf_level}" TYPE="normal" FOLDER="${directory}" PR="${pruning_rate}" DATA_FOLDER="${data_folder}" SAVE_FOLDER="${save_folder}"
  fi

#done

#done

}



for model in "resnet_small"; do
for dataset in "small_imagenet"; do
for pruning_rate in "0" "0.9"; do
for rf_level in "3" "4" "5" "6" "7" "8" "9" "10"; do


#run_saturation_calc "${model}" "${dataset}" "${HOME}/resnet_small_saturation" "${HOME}/datasets" "${HOME}/sparse_ensemble/saturation_results/small_imagenet/resnet_small/SGD" "no_recording_200" 0 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" "${pruning_rate}" "${rf_level}"


run_saturation_calc "${model}" "${dataset}" "${HOME}/resnet_small_saturation" "${HOME}/datasets" "${HOME}/sparse_ensemble/saturation_results/small_imagenet/resnet_small/SGD" "no_recording_200" 0 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" "${pruning_rate}" "${rf_level}"

done
done
done
done





#resolution=64
#resize=0
#save_folder="${HOME}/sparse_ensemble/dilation_results"
#checkpoints_folder="${SCRATCH}/dilation_experiments_100"
#for model in "vgg19_dilation" ; do
#for dataset in "tiny_imagenet"; do
##for dataset in "small_imagenet"; do
##for pruning_rate in "0.9" ; do
#for pruning_rate in "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8"; do
#for rf_level in 91 180 269; do
##for rf_level in 54 107 159 407; do    # iterate idxa from 0 to length
##for rf_level in "1"; do
##for rf_level in "5" "6" "7" "8" "10"; do
##for rf_level in "11" "12" "13"; do
##for rf_level in "1" "2" "3" "4"; do
#
#
##run_pruning "${model}" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_approx_200_no_ffcv" 0 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" "${pruning_rate}" "${rf_level}" "${resolution}" "${resize}"
#
#run_pruning "${model}" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_dilation_100_no_ffcv" 0 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" "${pruning_rate}" "${rf_level}" "${resolution}" "${resize}"
#
#
#done
#done
#done
#done





# ResNet50- CIFAR10

resolution=32
resize=0
#save_folder="${HOME}/sparse_ensemble/dilation_results"
save_folder="${HOME}/saturation_dilation_results/cifar10/resnet50"
checkpoints_folder="${SCRATCH}/dilation_experiments_100"
for model in "resnet50_dilation"; do
for dataset in "cifar10"; do
#for dataset in "small_imagenet"; do
#for pruning_rate in "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8"; do
for pruning_rate in "0" ;do
#for rf_level in 107 655 907 1497; do    # iterate idxa from 0 to length
for rf_level in 54 107 159 655 907 1497; do    # iterate idxa from 0 to length
#for rf_level in "5" "6" "7" "8" "10"; do
#for rf_level in "11" "12" "13"; do
#for rf_level in "1" "2" "3" "4"; do


#run_pruning "${model}" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_approx_200_no_ffcv" 0 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" "${pruning_rate}" "${rf_level}" "${resolution}" "${resize}"

#run_pruning "${model}" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_dilation_100_no_ffcv" 0 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" "${pruning_rate}" "${rf_level}" "${resolution}" "${resize}"

run_saturation_calc "${model}" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_dilation_100_no_ffcv" 0 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" "${pruning_rate}" "${rf_level}"

done
done
done
done



# ResNet50- Tiny ImageNet

resolution=64
resize=0
save_folder="${HOME}/saturation_dilation_results/cifar10/resnet50"
checkpoints_folder="${SCRATCH}/dilation_experiments_100"
for model in "resnet50_dilation"; do
for dataset in "tiny_imagenet"; do
#for dataset in "small_imagenet"; do
#for pruning_rate in "0.9" ; do
for rf_level in "54" "107" "159" "655" "907" "1497"; do
#for rf_level in "5" "6" "7" "8" "10"; do
#for rf_level in "11" "12" "13"; do
#for rf_level in "1" "2" "3" "4"; do


run_saturation_calc "${model}" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_dilation_100_no_ffcv" 0 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" "${pruning_rate}" "${rf_level}"



done
done
done
done

