#!/bin/bash
run_pruning() {
model=$1
dataset=$2
directory=$3
data_folder=$4
save_folder=$5
name=$6
ffcv=$7
ffcv_train=$8
ffcv_val=$9
#pruning_rates=("0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9")
#pruning_rates=("0.5")
pruning_rate="${10}"
rf_level="${11}"
resolution="${12}"
resize="${13}"
dataset2="${14}"
# For resnet18
#
#  if [ "${10}" -gt 0 ]
#  then
##      rf_levels=("5" "7" "8" "9")
#       rf_levels=("5")
#
#  else
#        rf_levels=("3" "4" "6" "10")
##         rf_levels=("6")
#  fi

#levels_max=${#rf_levels[@]}                                  # Take the length of that array
#number_pruning_rates=${#pruning_rates[@]}                            # Take the length of that array
#for ((idxA=0; idxA<number_pruning_rates; idxA++)); do                # iterate idxA from 0 to length
#for ((idxB=0; idxB<levels_max; idxB++));do              # iterate idxB from 0 to length


if [ "${ffcv}" -gt 0 ]
  then
  echo "Use FFCV"
sbatch --nodes=1 --time=48:00:00 --partition=gpu --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk  --error="${model}_${rf_level}_${dataset}_${pruning_rate]}_transfer_to_${dataset2}_ffcv.err" --output="${model}_${rf_level}_${dataset}_${pruning_rate}_transfer_to_${dataset2}_ffcv.out" --job-name="${model}_${rf_level}_${dataset}_${pruning_rate}_transfer_to_${dataset2}_ffcv" slurm_prune_finetune_another_task_run.sh FFCV="${ffcv}" NAME="${name}" MODEL="${model}" DATASET="${dataset}"  NUMW=8  RFL="${rf_level}" TYPE="normal" FOLDER="${directory}" PR="${pruning_rate}" EXPERIMENT=9 FFCV_TRAIN="${ffcv_train}" FFCV_VAL="${ffcv_val}" DATA_FOLDER="${data_folder}" SAVE_FOLDER="${save_folder}" INPUT_RES="${resolution}" RESIZE="${resize}" DATASET2"${dataset2}"

else
 echo "Don't use FFCV"
 sbatch --nodes=1 --time=48:00:00 --partition=gpu --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk  --error="${model}_${rf_level}_${dataset}_${pruning_rate}_transfer_to_${dataset2}_no_ffcv.err" --output="${model}_${rf_level}_${dataset}_${pruning_rate}_transfer_to_${dataset2}_no_ffcv.out" --job-name="${model}_${rf_level}_${dataset}_${pruning_rate}_transfer_to_${dataset2}_no_ffcv"  slurm_prune_finetune_another_task_run.sh FFCV="${ffcv}" NAME="${name}" MODEL="${model}" DATASET="${dataset}"  NUMW=8  RFL="${rf_level}" TYPE="normal" FOLDER="${directory}" PR="${pruning_rate}" EXPERIMENT=9 DATA_FOLDER="${data_folder}" SAVE_FOLDER="${save_folder}" INPUT_RES="${resolution}" RESIZE="${resize}" DATASET2="${dataset2}"

  fi
#done
#done

}
#=====================================================================================
#                 CIFAR10 => TINY IMAGENET
#=====================================================================================
###################################
#       Resnet50
###################################
resolution_dataset2=64
resolution_dataset1=32
resize=0
#resolution=224
#save_folder="${HOME}/sparse_ensemble/small_imagenet_resized_experiments_pruning"
#save_folder="${HOME}/sparse_ensemble/"
save_folder="/users/sclaam/sparse_ensemble/RF_transfer_experiments"
checkpoints_folder="${HOME}/checkpoints/original_paper_checkpoints"
dataset2="tiny_imagenet"
for model in "resnet50"; do
#for dataset in "cifar10"; do
for dataset in "cifar10"; do
for pruning_rate in "0.9"; do
for rf_level in "2" "4"; do
run_pruning "${model}" "${dataset}" "${checkpoints_folder}" "${HOME}/data2" "${save_folder}" "recording_200_no_ffcv" 0 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" "${pruning_rate}" "${rf_level}" "${resolution_dataset2}" "${resize}" "${dataset2}"
done
done
done
done

###################################
#       VGG19
###################################

save_folder="/users/sclaam/sparse_ensemble/RF_transfer_experiments"
checkpoints_folder="${HOME}/checkpoints/original_paper_checkpoints"
dataset2="tiny_imagenet"
for model in "vgg19"; do
#for dataset in "cifar10"; do
for dataset in "cifar10"; do
for pruning_rate in "0.9"; do
for rf_level in "1" "3"; do
run_pruning "${model}" "${dataset}" "${checkpoints_folder}" "${HOME}/data2" "${save_folder}" "recording_200_no_ffcv" 0 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" "${pruning_rate}" "${rf_level}" "${resolution_dataset2}" "${resize}" "${dataset2}"
done
done
done
done

#=====================================================================================
#                 TINY IMAGENET  => CIFAR10
#=====================================================================================

###################################
#       Resnet50
###################################
resolution_dataset2=32
resolution_dataset1=64
resize=0
#resolution=224
#save_folder="${HOME}/sparse_ensemble/small_imagenet_resized_experiments_pruning"
#save_folder="${HOME}/sparse_ensemble/"
save_folder="/users/sclaam/sparse_ensemble/RF_transfer_experiments"
checkpoints_folder="${HOME}/checkpoints/original_paper_checkpoints"
dataset2="cifar10"
for model in "resnet50"; do
#for dataset in "cifar10"; do
for dataset in "tiny_imagenet "; do
for pruning_rate in "0.9"; do
for rf_level in "2" "4"; do
run_pruning "${model}" "${dataset}" "${checkpoints_folder}" "${HOME}/data2" "${save_folder}" "recording_200_no_ffcv" 0 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" "${pruning_rate}" "${rf_level}" "${resolution_dataset2}" "${resize}" "${dataset2}"
done
done
done
done

###################################
#       VGG19
###################################

save_folder="/users/sclaam/sparse_ensemble/RF_transfer_experiments"
checkpoints_folder="${HOME}/checkpoints/original_paper_checkpoints"
dataset2="cifar10"
for model in "vgg19"; do
#for dataset in "cifar10"; do
for dataset in "tiny_imagenet"; do
for pruning_rate in "0.9"; do
for rf_level in "1" "3"; do
run_pruning "${model}" "${dataset}" "${checkpoints_folder}" "${HOME}/data2" "${save_folder}" "recording_200_no_ffcv" 0 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" "${pruning_rate}" "${rf_level}" "${resolution_dataset2}" "${resize}" "${dataset2}"
done
done
done
done
