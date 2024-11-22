#!/bin/bash
#### Command

#python3.9 prune_models.py --name "${name}" --model "${model}" --dataset "${dataset}" --num_workers 0 --RF_level "${rf_level}" --type "normal" --folder "${directory}" --pruning_rate "${pruning_rate}"  --experiment 8 --data_folder "${data_folder}" --save_folder "${save_folder}"

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

#  ./local_filter_quality_run.sh FFCV="${ffcv}" NAME="${name}" MODEL="${model}" DATASET="${dataset}"  NUMW=0  RFL="${rf_level}" TYPE="normal" FOLDER="${directory}" PR="${pruning_rate}" EXPERIMENT=5 FFCV_TRAIN="${ffcv_train}" FFCV_VAL="${ffcv_val}" DATA_FOLDER="${data_folder}" SAVE_FOLDER="${save_folder}"

python3.9 prune_models.py --ffcv --name "${name}" --model "${model}" --dataset "${dataset}" --num_workers 0 --RF_level "${rf_level}" --type "normal" --folder "${directory}" --pruning_rate "${pruning_rate}"  --experiment 6 --data_folder "${data_folder}" --save_folder "${save_folder}" --input_resolution  "${resolution}" --resize  "${resize}"


else

 echo "Don't use FFCV"
#./local_filter_quality_run.sh

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

qsub -l h_rt=44:00:00 -t 1-3 -l coproc_v100=1  -N  "calculate_saturation_${model}_${dataset}_${rf_level}_${name}" arc4_calculate_saturation.sh "${model}" "${dataset}" "${directory}" "${data_folder}" "${save_folder}" "${name}" "${ffcv}" "${ffcv_train}" "${ffcv_val}" "${pruning_rate}" "${rf_level}" "${resolution}" "${resize}"

#  FFCV="${ffcv}" NAME="${name}" MODEL="${model}" DATASET="${dataset}"  NUMW=0  RFL="${rf_level}" TYPE="normal" FOLDER="${directory}" PR="${pruning_rate}" EXPERIMENT=5 DATA_FOLDER="${data_folder}" SAVE_FOLDER="${save_folder}"

  fi

#done
#done

}


#solutions_folder="/home/luisaam/Downloads/resnet_25_small_imagenet"
solutions_folder="/nobackup/sclaam/resnet_25_small_imagenet"
#solutions_folder="/home/luisaam/Downloads/deep_small_models_resized"
#data_folder="/home/luisaam/Documents/PhD/data/"
data_folder="/nobackup/sclaam/data/"
#save_folder="/home/luisaam/PycharmProjects/sparse_ensemble"
#save_folder="/home/luisaam/PycharmProjects/sparse_ensemble/filter_quality_results/small_imagenet/"
#save_folder="/home/luisaam/PycharmProjects/sparse_ensemble/filter_quality_results/small_imagenet_resized/"
#save_folder="/home/luisaam/PycharmProjects/sparse_ensemble/saturation_results/small_imagenet/resnet25_small"
save_folder="${HOME}/sparse_ensemble/saturation_results/small_imagenet/resnet25_small"
#save_folder="/home/luisaam/PycharmProjects/sparse_ensemble/saturation_results/small_imagenet_resized/resnet25_small"

#resolution=224

resolution=224
resize=0
for model in "resnet25_small"; do
for dataset in "small_imagenet"; do
for pruning_rate in "0"; do
for rf_level in "5" "6" "7" "8" "10" "11" "12" "13"; do
#for rf_level in "6" ""; do

run_pruning "${model}" "$dataset" "${solutions_folder}" "${data_folder}" "${save_folder}" "sgd_100_res_224_no_ffcv" 0 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" "${pruning_rate}" "${rf_level}" "${resolution}" "${resize}"

done
done
done
done
#solutions_folder="/home/luisaam/Downloads/resnet_25_small_imagenet"
solutions_folder="/home/luisaam/Downloads/deep_small_models_resized"
data_folder="/home/luisaam/Documents/PhD/data/"
#save_folder="/home/luisaam/PycharmProjects/sparse_ensemble"
#save_folder="/home/luisaam/PycharmProjects/sparse_ensemble/filter_quality_results/small_imagenet/"
#save_folder="/home/luisaam/PycharmProjects/sparse_ensemble/filter_quality_results/small_imagenet_resized/"
#save_folder="/home/luisaam/PycharmProjects/sparse_ensemble/saturation_results/small_imagenet/resnet25_small"
save_folder="/home/luisaam/PycharmProjects/sparse_ensemble/saturation_results/small_imagenet_resized/resnet25_small"

resolution=224
resize=1

for model in "resnet25_small"; do
for dataset in "small_imagenet"; do
for pruning_rate in "0"; do
for rf_level in "5" "6" "7" "8" "10" "11" "12" "13"; do
#for rf_level in "6" ""; do

run_pruning "${model}" "$dataset" "${solutions_folder}" "${data_folder}" "${save_folder}" "sgd_100_res_224_no_ffcv" 0 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" "${pruning_rate}" "${rf_level}" "${resolution}" "${resize}"

done
done
done
done
