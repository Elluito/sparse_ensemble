#!/bin/bash
run_sp_fine_tuning() {
model=$1

dataset=$2
#directory=$4
#ffcv=$6
#ffcv_train=$7
#ffcv_val=$8
sigma=$3
pr=$4
pruner=$5
resolution=$6
name=$7
#pruning_rates=("0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9")
#pruning_rates=("0.5")
# For resnet18
#
#  if [ "${10}" -gt 0 ]




#sbatch --nodes=1 --time=140:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --gres=gpu:1 --error="training_probes_${model}_${dataset}_rf_level_${rf_level}.err"  --output="training_probes_${model}_${dataset}_rf_level_${rf_level}.out"  --job-name="training_probes_${model}_${dataset}_rf_level_${rf_level}" slurm_training_probes_run.sh "${model}" "${dataset}" "${rf_level}" "${resolution}" "${save_folder}" "${name}"
#qsub -l h_rt=48:00:00 -t 1-5 -l coproc_v100=1 -N "SP_FT_${model}_${dataset}_sig_${sigma}_pr_${pr}_sto" arc4_SP_fine_tuning_run.sh 11 "${sigma}" "${pruner}" "${model}" "${dataset}" "${pr}" "alternative" 100 "${name}" #"${save_folder}" "${name}"

qsub -l h_rt=48:00:00 -l coproc_v100=1  -N "SP_FT_${model}_${dataset}_sig_${sigma}_pr_${pr}_det" arc4_SP_fine_tuning_run.sh 6 "${sigma}" "${pruner}" "${model}" "${dataset}" "${pr}"  "alternative" 100 "${name}" #"${save_folder}" "${name}"

}

###############TPE side of the table#######################
#model_list=("resnet18" "resnet18" "resnet50" "resnet50" "vgg19" "vgg19")
#
#dataset_list=("cifar10" "cifar100" "cifar10" "cifar100" "cifar10" "cifar100")
#
#sigma_list=("0.00456" "0.00485" "0.00256" "0.00194" "0.00110" "0.00184")
#
#pruning_rate_list=("0.86" "0.92" "0.94" "0.77" "0.91" "0.81")
#
#max=${#model_list[@]}                                  # Take the length of that array
#for ((idxA=0; idxA<max; idxA++)); do              # iterate idxA from 0 to length
#model="${model_list[$idxA]}"
#dataset="${dataset_list[$idxA]}"
#sigma="${sigma_list[$idxA]}"
#pruning_rate="${pruning_rate_list[$idxA]}"
#
#run_sp_fine_tuning "${model}" "${dataset}" "${sigma}" "${pruning_rate}" "global" "32" "tpe" #"${HOME}/second_order_saturation" "${HOME}/datasets" "${HOME}/sparse_ensemble/second_order_pruning" "sam_optim_saturation_200_gc_0" 0 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" "${pruning_rate}" "${rf_level}"
#done
#
############### NSGA-II side of the table#######################
#model_list=("resnet18" "resnet18" "resnet50" "resnet50" "vgg19" "vgg19")
#dataset_list=("cifar10" "cifar100" "cifar10" "cifar100" "cifar10" "cifar100")
#sigma_list=("0.0038" "0.0036" "0.0028" "0.0012" "0.0013" "0.0025")
#pruning_rate_list=("0.8742" "0.92" "0.948" "0.76" "0.915" "0.83")
#
#max=${#model_list[@]}                                  # Take the length of that array
#for ((idxA=0; idxA<max; idxA++)); do              # iterate idxA from 0 to length
#model="${model_list[$idxA]}"
#dataset="${dataset_list[$idxA]}"
#sigma="${sigma_list[$idxA]}"
#pruning_rate="${pruning_rate_list[$idxA]}"
#
#run_sp_fine_tuning "${model}" "${dataset}" "${sigma}" "${pruning_rate}" "global" "32" "nsga" #"${HOME}/second_order_saturation" "${HOME}/datasets" "${HOME}/sparse_ensemble/second_order_pruning" "sam_optim_saturation_200_gc_0" 0 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" "${pruning_rate}" "${rf_level}"
#
#done





############## Re-run of gradientflow and fine-tuned accuracy plots #######################

model_list=("resnet18" "resnet18" "resnet50" "resnet50" "vgg19" "vgg19")

dataset_list=("cifar10" "cifar100" "cifar10" "cifar100" "cifar10" "cifar100")

sigma_list=("0.005" "0.005" "0.005" "0.005" "0.005" "0.005")

pruning_rate_list=("0.9" "0.9" "0.95" "0.85" "0.95" "0.8")

max=${#model_list[@]}                                  # Take the length of that array
for ((idxA=0; idxA<max; idxA++)); do              # iterate idxA from 0 to length
model="${model_list[$idxA]}"
dataset="${dataset_list[$idxA]}"
sigma="${sigma_list[$idxA]}"
pruning_rate="${pruning_rate_list[$idxA]}"

run_sp_fine_tuning "${model}" "${dataset}" "${sigma}" "${pruning_rate}" "global" "32" "FT" #"${HOME}/second_order_saturation" "${HOME}/datasets" "${HOME}/sparse_ensemble/second_order_pruning" "sam_optim_saturation_200_gc_0" 0 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" "${pruning_rate}" "${rf_level}"
done

############## NSGA-II side of the table#######################
model_list=("resnet18" "resnet18" "resnet50" "resnet50" "vgg19" "vgg19")
dataset_list=("cifar10" "cifar100" "cifar10" "cifar100" "cifar10" "cifar100")
sigma_list=("0.001" "0.001" "0.001" "0.001" "0.001" "0.001")

pruning_rate_list=("0.9" "0.9" "0.95" "0.85" "0.95" "0.8")

max=${#model_list[@]}                                  # Take the length of that array
for ((idxA=0; idxA<max; idxA++)); do              # iterate idxA from 0 to length
model="${model_list[$idxA]}"
dataset="${dataset_list[$idxA]}"
sigma="${sigma_list[$idxA]}"
pruning_rate="${pruning_rate_list[$idxA]}"

run_sp_fine_tuning "${model}" "${dataset}" "${sigma}" "${pruning_rate}" "global" "32" "FT" #"${HOME}/second_order_saturation" "${HOME}/datasets" "${HOME}/sparse_ensemble/second_order_pruning" "sam_optim_saturation_200_gc_0" 0 "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" "${pruning_rate}" "${rf_level}"

done
