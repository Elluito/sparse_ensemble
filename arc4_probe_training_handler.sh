#!/bin/bash
run_probe_training() {
model=$1

dataset=$2
#directory=$4
#ffcv=$6
#ffcv_train=$7
#ffcv_val=$8
rf_level=$3
resolution=$4
save_folder=$5
name=$6
#pruning_rates=("0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9")
#pruning_rates=("0.5")
# For resnet18
#
#  if [ "${10}" -gt 0 ]




#sbatch --nodes=1 --time=140:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --gres=gpu:1 --error="training_probes_${model}_${dataset}_rf_level_${rf_level}.err"  --output="training_probes_${model}_${dataset}_rf_level_${rf_level}.out"  --job-name="training_probes_${model}_${dataset}_rf_level_${rf_level}" slurm_training_probes_run.sh "${model}" "${dataset}" "${rf_level}" "${resolution}" "${save_folder}" "${name}"
qsub -l h_rt=48:00:00 -N "training_probes_${model}_${dataset}_rf_level_${rf_level}_${name}"  arc4_probe_training_run.sh "${model}" "${dataset}" "${rf_level}" "${resolution}" "${save_folder}" "${name}"

}




logs_folder="./probes_logs/"



#
#
#rf_levels=("1" "2" "3" "4")
#levels_max=${#rf_levels[@]}                                  # Take the length of that array
#for ((idxA=0; idxA<levels_max; idxA++));do              # iterate idxB from 0 to length
# run_probe_training "vgg19" "cifar10" "${rf_levels[$idxA]}" "224" "${logs_folder}" "x" # "${names[$idxA]}"  "${save_folder}" "50" "0.001" "0"
#done
#
#
#
#rf_levels=("1" "2" "3" "4" "9" "10" "11")
#levels_max=${#rf_levels[@]}                                  # Take the length of that array
#for ((idxA=0; idxA<levels_max; idxA++));do              # iterate idxB from 0 to length
# run_probe_training "resnet50" "cifar10" "${rf_levels[$idxA]}" "32" "${logs_folder}" "x"   # "${names[$idxA]}"  "${save_folder}" "50" "0.001" "0"
#done

rf_levels=("5" "6" "7" "8" "10" "11" "12" "13")
levels_max=${#rf_levels[@]}                                  # Take the length of that array
for ((idxA=0; idxA<levels_max; idxA++));do              # iterate idxB from 0 to length
 run_probe_training "resnet25_small" "small_imagenet" "${rf_levels[$idxA]}" "224" "${logs_folder}" "sgd_100_res_224_no_ffcv_test_pr_0.95_adjust_bn"   # "${names[$idxA]}"  "${save_folder}" "50" "0.001" "0"

done
