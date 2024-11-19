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
#pruning_rates=("0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9")
#pruning_rates=("0.5")
# For resnet18
#
#  if [ "${10}" -gt 0 ]




qsub -l h_rt=47:00:00 -l coproc_v100=1  -N "extract_features_${model}_${dataset}_rf_level_${rf_level}" slurm_probes_extraction_run.sh "${data_folder}" "${model}" "normal" "${solution}" "${dataset}" "${rf_level}" "${resolution}" "${num_workers}" "${batch_size}" "${save_folder}" "${name}"

}

data_folder="/nobackup/sclaam/data"
logs_folder="./probes_logs/"





model="resnet25_small"
dataset="small_imagenet"
resolution=224

sol1rs25="/nobackup/sclaam/deep_small_models_2/"
sol2rs25="/nobackup/sclaam/deep_small_models_2/"
sol3rs25="/nobackup/sclaam/deep_small_models_2/"
sol4rs25="/nobackup/sclaam/deep_small_models_2/"
sol5rs25="/nobackup/sclaam/deep_small_models_2/"
sol6rs25="/nobackup/sclaam/deep_small_models_2/"
sol7rs25="/nobackup/sclaam/deep_small_models_2/"
sol8rs25="/nobackup/sclaam/deep_small_models_2/"



rf_levels=("1" "2" "3" "4" "9" "10" "11")
solutions=("${sol1rs25}" "${sol2rs25}" "${sol3rs25}" "${sol4rs25}" "${sol5rs25}")
names=("seed_0" "seed_0" "seed_0" "seed_0" "seed_0" "seed_0" "seed_0")
#names=("seed_0" "seed_0" "seed_0" "seed_0" "seed_0" "seed_0" "seed_0")
levels_max=${#rf_levels[@]}                                  # Take the length of that array
for ((idxA=0; idxA<levels_max; idxA++));do              # iterate idxB from 0 to length
 run_extraction "${model}" "${dataset}" "${data_folder}" "${solutions[$idxA]}" "${rf_levels[$idxA]}" "${resolution}" 2 128 "${logs_folder}" "sgd" # "${names[$idxA]}"  "${save_folder}" "50" "0.001" "0"
done


