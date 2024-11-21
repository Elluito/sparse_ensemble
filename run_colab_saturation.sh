#!/bin/bash

run_pruning(){
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
resolution="${12}"
resize="${13}"
if [ "${ffcv}" -gt 0 ]
  then
  echo "Use FFCV"
#  ./local_filter_quality_run.sh FFCV="${ffcv}" NAME="${name}" MODEL="${model}" DATASET="${dataset}"  NUMW=0  RFL="${rf_level}" TYPE="normal" FOLDER="${directory}" PR="${pruning_rate}" EXPERIMENT=5 FFCV_TRAIN="${ffcv_train}" FFCV_VAL="${ffcv_val}" DATA_FOLDER="${data_folder}" SAVE_FOLDER="${save_folder}"
python3.9 sparse_ensemble/prune_models.py --ffcv --name "${name}" --model "${model}" --dataset "${dataset}" --num_workers 0 --RF_level "${rf_level}" --type "normal" --folder "${directory}" --pruning_rate "${pruning_rate}"  --experiment 6 --data_folder "${data_folder}" --save_folder "${save_folder}"
else
 echo "Don't use FFCV"
python3.9 sparse_ensemble/prune_models.py --name "${name}" --model "${model}" --dataset "${dataset}" --num_workers 0 --RF_level "${rf_level}" --type "normal" --folder "${directory}" --pruning_rate "${pruning_rate}"  --experiment 6 --data_folder "${data_folder}" --save_folder "${save_folder}" --resize "${resize}" --input_resolution "${resolution}" --resize "${resize}"
fi
}
solutions_folder="/content/drive/MyDrive/PhD/solutions_small_imagenet/resnet_25_small_imagenet"
#solutions_folder="/content/drive/MyDrive/PhD/solutions_small_imagenet/deep_small_models_resized"
data_folder="/home/luisaam/Documents/PhD/data/"
#save_folder="/home/luisaam/PycharmProjects/sparse_ensemble"
#save_folder="/home/luisaam/PycharmProjects/sparse_ensemble/filter_quality_results/small_imagenet/"
#save_folder="/home/luisaam/PycharmProjects/sparse_ensemble/filter_quality_results/small_imagenet_resized/"
save_folder="sparse_ensemble/saturation_results/small_imagenet/resnet25_small"
#save_folder="sparse_ensemble/saturation_results/small_imagenet_resized/resnet25_small"
#resolution=224
for model in "resnet25_small"; do
for dataset in "small_imagenet"; do
for pruning_rate in "0"; do
for rf_level in "5" "6" "7" "8" "10" "11" "12" "13"; do
run_pruning "${model}" "$dataset" "${solutions_folder}" "${data_folder}" "${save_folder}" "sgd_100_res_224_no_ffcv" 0 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" "${pruning_rate}" "${rf_level}" 224 0
done
done
done
done
