#!/bin/bash


run_statistics() {
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
sbatch --nodes=1 --time=03:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk  --error="${model}_${rf_level}_${dataset}_${pruning_rate]}_weight_distributions_ffcv.err" --output="${model}_${rf_level}_${dataset}_${pruning_rate}_weight_distributions_ffcv.out" --job-name="${model}_${rf_level}_${dataset}_${pruning_rate}_weight_distributions_ffcv"   slurm_pruning_run.sh FFCV="${ffcv}" NAME="${name}" MODEL="${model}" DATASET="${dataset}"  NUMW=0  RFL="${rf_level}" TYPE="normal" FOLDER="${directory}" PR="${pruning_rate}" EXPERIMENT=7 FFCV_TRAIN="${ffcv_train}" FFCV_VAL="${ffcv_val}" DATA_FOLDER="${data_folder}" SAVE_FOLDER="${save_folder}" INPUT_RES="${resolution}"

else
 echo "Don't use FFCV"
 sbatch --nodes=1 --time=03:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk  --error="${model}_${rf_level}_${dataset}_${pruning_rate}_weight_distributions_no_ffcv.err" --output="${model}_${rf_level}_${dataset}_${pruning_rate}_weight_distributions_no_ffcv.out" --job-name="${model}_${rf_level}_${dataset}_${pruning_rate}_weight_distributions_no_ffcv" slurm_pruning_run.sh FFCV="${ffcv}" NAME="${name}" MODEL="${model}" DATASET="${dataset}"  NUMW=0  RFL="${rf_level}" TYPE="normal" FOLDER="${directory}" PR="${pruning_rate}" EXPERIMENT=7 DATA_FOLDER="${data_folder}" SAVE_FOLDER="${save_folder}" INPUT_RES="${resolution}"
  fi
#done
#done

}

# New models pruning results
#for model in "densenet40"; do
#for dataset in "tiny_imagenet"; do
#for pruning_rate in "0.8" "0.9" ; do
#for rf_level in "4" "8"; do
#
#run_pruning "${model}" "$dataset" "${HOME}/new_models_original_paper" "${HOME}/datasets" "${HOME}/sparse_ensemble/new_models_original_paper" "recording_200" 0 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" "${pruning_rate}" "${rf_level}"
#
#done
#done
#done
#done

# For EKFAC
#for model in "vgg19"; do
#for dataset in "cifar10"; do
#for pruning_rate in "0.7" "0.8" "0.9" "0.95"; do
#for rf_level in "3" "4"; do
#
#run_pruning "${model}" "$dataset" "${HOME}/second_order_experiments" "${HOME}/datasets" "${HOME}/sparse_ensemble/second_order_pruning" "ekfac_optim_hyper_saturation_200_gc_0.4" 0 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" "${pruning_rate}" "${rf_level}"
#
#done
#done
#done
#done

# For original paper

#for model in "resnet50"; do
#for dataset in "cifar10"; do
#for pruning_rate in "0.9" "0.95"; do
#for rf_level in "1" "2" "3" "4"; do
#
#run_pruning "${model}" "$dataset" "${HOME}/second_order_saturation" "${HOME}/datasets" "${HOME}/sparse_ensemble/second_order_pruning" "ekfac_optim_hyper_saturation_200_gc_0" 0 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" "${pruning_rate}" "${rf_level}"
#
#done
#done
#done
#done

#for model in "vgg19"; do
#for dataset in "cifar10"; do
#for pruning_rate in "0.8" "0.9" "0.95"; do
#for rf_level in "1" "2" "3" "4"; do
#
#run_pruning "${model}" "$dataset" "${HOME}/second_order_saturation" "${HOME}/datasets" "${HOME}/sparse_ensemble/second_order_pruning" "sam_optim_saturation_200_gc_0" 0 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" "${pruning_rate}" "${rf_level}"
#
#done
#done
#done
#done
resolution=32
#
#for model in "resnet25_small"; do
#for dataset in "cifar10"; do
#for pruning_rate in "0.9"; do
#for rf_level in "5" "6" "7" "8" "10"; do
#
##run_saturation_calc "${model}" "${dataset}" "${HOME}/resnet_small_saturation" "${HOME}/datasets" "${HOME}/sparse_ensemble/saturation_results/small_imagenet/resnet_small/SGD" "no_recording_200" 0 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" "${pruning_rate}" "${rf_level}"
#
#run_pruning "${model}" "${dataset}" "${HOME}/deep_small_models_saturation" "${HOME}/datasets" "${HOME}/sparse_ensemble/large_input_pruning_results" "sgd_100_res_224" 0 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" "${pruning_rate}" "${rf_level}" "${resolution}"
#run_pruning "${model}" "${dataset}" "${HOME}/deep_small_models_saturation" "${HOME}/datasets" "${HOME}/sparse_ensemble/large_input_pruning_results" "ekfac_cifar10_100_res_224_gc_0" 0 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" "${pruning_rate}" "${rf_level}" "${resolution}"
#run_pruning "${model}" "${dataset}" "${HOME}/deep_small_models_saturation" "${HOME}/datasets" "${HOME}/sparse_ensemble/large_input_pruning_results" "sam_cifar10_100_res_224_gc_0" 0 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" "${pruning_rate}" "${rf_level}" "${resolution}"
#
#
#done
#done
#done
#done

resolution=32
for model in "resnet50"; do
for dataset in "cifar10"; do
for pruning_rate in "0"; do
for rf_level in "1" "2" "3" "4"; do


run_statistics "${model}" "$dataset" "${HOME}/deep_small_models_saturation" "${HOME}/datasets" "${HOME}/sparse_ensemble/weight_distributions" "sgd_200_res_32_lr_0.01" 0 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" "${pruning_rate}" "${rf_level}" "${resolution}"
run_statistics "${model}" "$dataset" "${HOME}/deep_small_models_saturation" "${HOME}/datasets" "${HOME}/sparse_ensemble/weight_distributions" "lr_changed_sam_200_res_32_gc_0" 0 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" "${pruning_rate}" "${rf_level}" "${resolution}"
run_statistics "${model}" "$dataset" "${HOME}/deep_small_models_saturation" "${HOME}/datasets" "${HOME}/sparse_ensemble/weight_distributions" "lr_changed_ekfac_200_res_32_gc_0" 0 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" "${pruning_rate}" "${rf_level}" "${resolution}"


done
done
done
done
