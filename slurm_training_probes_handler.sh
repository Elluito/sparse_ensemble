#!/bin/bash

run_training() {
model=$1
dataset=$2
#directory=$4
#ffcv=$6
#ffcv_train=$7
#ffcv_val=$8
rf_level=$3
resolution=$4
save_folder=$5
#pruning_rates=("0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9")
#pruning_rates=("0.5")
# For resnet18
#
#  if [ "${10}" -gt 0 ]




sbatch --nodes=1 --time=140:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --gres=gpu:1 --error="training_probes_${model}_${dataset}_rf_level_${rf_level}.err"  --output="training_probes_${model}_${dataset}_rf_level_${rf_level}.out"  --job-name="training_probes_${model}_${dataset}_rf_level_${rf_level}" slurm_probes_extraction_run.sh "${model}" "${dataset}" "${rf_level}" "${resolution}" "${save_folder}"

}




logs_folder="./probes_logs/"





rf_levels=("1" "2" "3" "4")
levels_max=${#rf_levels[@]}                                  # Take the length of that array
for ((idxA=0; idxA<levels_max; idxA++));do              # iterate idxB from 0 to length
 run_extraction "vgg19" "cifar10" "${rf_levels[$idxA]}" "32" "${logs_folder}"  # "${names[$idxA]}"  "${save_folder}" "50" "0.001" "0"
done



rf_levels=("1" "2" "3" "4" "9" "10" "11")
levels_max=${#rf_levels[@]}                                  # Take the length of that array
for ((idxA=0; idxA<levels_max; idxA++));do              # iterate idxB from 0 to length
 run_extraction "resnet50" "cifar10" "${rf_levels[$idxA]}" "32" "${logs_folder}"  # "${names[$idxA]}"  "${save_folder}" "50" "0.001" "0"
done

