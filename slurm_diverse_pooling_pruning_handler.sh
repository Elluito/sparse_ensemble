#!/bin/bash

# Pruning + fine-tuning handler for the diverse pooling models
# (spectral_pool, softpool, mixedpool, lippool) on resnet50/vgg19, cifar10, rf_level 1-4.
#
# Two experiments are launched via prune_models.py:
#   - experiment=1: one-shot pruning. Globs every *_test_acc_*.pth solution matching
#     model/dataset/rf_level inside ${directory} and prunes each one (no fine-tuning).
#   - experiment=2: prune + fine-tune (10 epochs, hardcoded in slurm_diverse_pooling_pruning_run.sh)
#     a single named solution passed via SOLUTION.

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
pruning_rate="${10}"
rf_level="${11}"
resolution="${12}"
resize="${13}"

if [ "${ffcv}" -gt 0 ]
then
  echo "Use FFCV"
sbatch --nodes=1 --time=48:00:00 --partition=gpu --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="${model}_${rf_level}_${dataset}_${pruning_rate}_pruning_ffcv.err" --output="${model}_${rf_level}_${dataset}_${pruning_rate}_pruning_ffcv.out" --job-name="${model}_${rf_level}_${dataset}_${pruning_rate}_pruning_ffcv" slurm_diverse_pooling_pruning_run.sh FFCV="${ffcv}" NAME="${name}" MODEL="${model}" DATASET="${dataset}" NUMW=8 RFL="${rf_level}" TYPE="normal" FOLDER="${directory}" PR="${pruning_rate}" EXPERIMENT=1 SOLUTION="" FFCV_TRAIN="${ffcv_train}" FFCV_VAL="${ffcv_val}" DATA_FOLDER="${data_folder}" SAVE_FOLDER="${save_folder}" INPUT_RES="${resolution}" RESIZE="${resize}"

else
 echo "Don't use FFCV"
 sbatch --nodes=1 --time=48:00:00 --partition=gpu --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="${model}_${rf_level}_${dataset}_${pruning_rate}_pruning_no_ffcv.err" --output="${model}_${rf_level}_${dataset}_${pruning_rate}_pruning_no_ffcv.out" --job-name="${model}_${rf_level}_${dataset}_${pruning_rate}_pruning_no_ffcv" slurm_diverse_pooling_pruning_run.sh FFCV="${ffcv}" NAME="${name}" MODEL="${model}" DATASET="${dataset}" NUMW=8 RFL="${rf_level}" TYPE="normal" FOLDER="${directory}" PR="${pruning_rate}" EXPERIMENT=1 SOLUTION="" DATA_FOLDER="${data_folder}" SAVE_FOLDER="${save_folder}" INPUT_RES="${resolution}" RESIZE="${resize}"
  fi
}

run_pruning_finetune() {
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
solution="${14}"

if [ "${ffcv}" -gt 0 ]
then
  echo "Use FFCV"
sbatch --nodes=1 --time=03:00:00 --partition=gpu --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="${model}_${rf_level}_${dataset}_${pruning_rate}_finetune_ffcv.err" --output="${model}_${rf_level}_${dataset}_${pruning_rate}_finetune_ffcv.out" --job-name="${model}_${rf_level}_${dataset}_${pruning_rate}_finetune_ffcv" slurm_diverse_pooling_pruning_run.sh FFCV="${ffcv}" NAME="${name}" MODEL="${model}" DATASET="${dataset}" NUMW=8 RFL="${rf_level}" TYPE="normal" FOLDER="${directory}" PR="${pruning_rate}" EXPERIMENT=2 SOLUTION="${solution}" FFCV_TRAIN="${ffcv_train}" FFCV_VAL="${ffcv_val}" DATA_FOLDER="${data_folder}" SAVE_FOLDER="${save_folder}" INPUT_RES="${resolution}" RESIZE="${resize}"

else
 echo "Don't use FFCV"
 sbatch --nodes=1 --time=03:00:00 --partition=gpu --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="${model}_${rf_level}_${dataset}_${pruning_rate}_finetune_no_ffcv.err" --output="${model}_${rf_level}_${dataset}_${pruning_rate}_finetune_no_ffcv.out" --job-name="${model}_${rf_level}_${dataset}_${pruning_rate}_finetune_no_ffcv" slurm_diverse_pooling_pruning_run.sh FFCV="${ffcv}" NAME="${name}" MODEL="${model}" DATASET="${dataset}" NUMW=8 RFL="${rf_level}" TYPE="normal" FOLDER="${directory}" PR="${pruning_rate}" EXPERIMENT=2 SOLUTION="${solution}" DATA_FOLDER="${data_folder}" SAVE_FOLDER="${save_folder}" INPUT_RES="${resolution}" RESIZE="${resize}"
  fi
}

#############################################################
# Experiment 1: one-shot pruning (all seeds found via glob in the checkpoints folder)
#############################################################

resolution=32
resize=0
save_folder="${HOME}/sparse_ensemble/diverse_pooling_pruning_results"
checkpoints_folder="${SCRATCH}/diverse_pooling_100"

for lvl in 1 2 3 4; do    # iterate rf_level from 1 to 4
for model in "resnet50_spectral_pool" "resnet50_softpool" "resnet50_mixedpool" "resnet50_lippool" "vgg19_spectral_pool" "vgg19_softpool" "vgg19_mixedpool" "vgg19_lippool"; do
for dataset in "cifar10"; do
for pruning_rate in "0.7" "0.8" "0.9"; do

run_pruning "${model}" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" "${lvl}" "${resolution}" "${resize}"

done
done
done
done


#############################################################
# Experiment 2: prune + fine-tune (10 epochs) a single named solution,
# one job per rf_level x model x dataset x seed (3 seeds per combo).
#
# One seed is intentionally omitted: vgg19_softpool rf_level_3, timestamp
# 1784642276.214949, finished at test_acc=10.0 (a collapsed/failed training
# run, not worth pruning+fine-tuning).
#############################################################

pruning_rate=0.9

for dataset in "cifar10"; do

# resnet50_spectral_pool
run_pruning_finetune "resnet50_spectral_pool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_spectral_pool_normal_cifar10_1784647877.4484823_rf_level_1_recording_diverse_pooling_100_no_ffcv_test_acc_94.9.pth"
run_pruning_finetune "resnet50_spectral_pool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_spectral_pool_normal_cifar10_1784649220.7222323_rf_level_1_recording_diverse_pooling_100_no_ffcv_test_acc_95.05.pth"
run_pruning_finetune "resnet50_spectral_pool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_spectral_pool_normal_cifar10_1784649289.176981_rf_level_1_recording_diverse_pooling_100_no_ffcv_test_acc_94.94.pth"
run_pruning_finetune "resnet50_spectral_pool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 2 "${resolution}" "${resize}" "resnet50_spectral_pool_normal_cifar10_1784672277.532727_rf_level_2_recording_diverse_pooling_100_no_ffcv_test_acc_92.17.pth"
run_pruning_finetune "resnet50_spectral_pool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 2 "${resolution}" "${resize}" "resnet50_spectral_pool_normal_cifar10_1784673024.8768005_rf_level_2_recording_diverse_pooling_100_no_ffcv_test_acc_92.87.pth"
run_pruning_finetune "resnet50_spectral_pool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 2 "${resolution}" "${resize}" "resnet50_spectral_pool_normal_cifar10_1784673273.9302685_rf_level_2_recording_diverse_pooling_100_no_ffcv_test_acc_92.23.pth"
run_pruning_finetune "resnet50_spectral_pool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 3 "${resolution}" "${resize}" "resnet50_spectral_pool_normal_cifar10_1784681120.1199763_rf_level_3_recording_diverse_pooling_100_no_ffcv_test_acc_92.19.pth"
run_pruning_finetune "resnet50_spectral_pool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 3 "${resolution}" "${resize}" "resnet50_spectral_pool_normal_cifar10_1784681299.5858653_rf_level_3_recording_diverse_pooling_100_no_ffcv_test_acc_91.77.pth"
run_pruning_finetune "resnet50_spectral_pool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 3 "${resolution}" "${resize}" "resnet50_spectral_pool_normal_cifar10_1784681604.6333723_rf_level_3_recording_diverse_pooling_100_no_ffcv_test_acc_92.01.pth"
run_pruning_finetune "resnet50_spectral_pool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_spectral_pool_normal_cifar10_1784688377.2915893_rf_level_4_recording_diverse_pooling_100_no_ffcv_test_acc_88.81.pth"
run_pruning_finetune "resnet50_spectral_pool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_spectral_pool_normal_cifar10_1784690334.121476_rf_level_4_recording_diverse_pooling_100_no_ffcv_test_acc_89.42.pth"
run_pruning_finetune "resnet50_spectral_pool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_spectral_pool_normal_cifar10_1784690395.6416922_rf_level_4_recording_diverse_pooling_100_no_ffcv_test_acc_89.43.pth"

# resnet50_softpool
run_pruning_finetune "resnet50_softpool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_softpool_normal_cifar10_1784649479.4396226_rf_level_1_recording_diverse_pooling_100_no_ffcv_test_acc_94.62.pth"
run_pruning_finetune "resnet50_softpool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_softpool_normal_cifar10_1784649691.350335_rf_level_1_recording_diverse_pooling_100_no_ffcv_test_acc_94.42.pth"
run_pruning_finetune "resnet50_softpool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_softpool_normal_cifar10_1784666044.038753_rf_level_1_recording_diverse_pooling_100_no_ffcv_test_acc_94.12.pth"
run_pruning_finetune "resnet50_softpool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 2 "${resolution}" "${resize}" "resnet50_softpool_normal_cifar10_1784673517.6280787_rf_level_2_recording_diverse_pooling_100_no_ffcv_test_acc_93.62.pth"
run_pruning_finetune "resnet50_softpool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 2 "${resolution}" "${resize}" "resnet50_softpool_normal_cifar10_1784673641.9483094_rf_level_2_recording_diverse_pooling_100_no_ffcv_test_acc_93.29.pth"
run_pruning_finetune "resnet50_softpool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 2 "${resolution}" "${resize}" "resnet50_softpool_normal_cifar10_1784673642.0665169_rf_level_2_recording_diverse_pooling_100_no_ffcv_test_acc_92.92.pth"
run_pruning_finetune "resnet50_softpool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 3 "${resolution}" "${resize}" "resnet50_softpool_normal_cifar10_1784684470.525072_rf_level_3_recording_diverse_pooling_100_no_ffcv_test_acc_91.29.pth"
run_pruning_finetune "resnet50_softpool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 3 "${resolution}" "${resize}" "resnet50_softpool_normal_cifar10_1784687459.2824476_rf_level_3_recording_diverse_pooling_100_no_ffcv_test_acc_91.15.pth"
run_pruning_finetune "resnet50_softpool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 3 "${resolution}" "${resize}" "resnet50_softpool_normal_cifar10_1784687581.8962588_rf_level_3_recording_diverse_pooling_100_no_ffcv_test_acc_90.79.pth"
run_pruning_finetune "resnet50_softpool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_softpool_normal_cifar10_1784692654.5222259_rf_level_4_recording_diverse_pooling_100_no_ffcv_test_acc_89.22.pth"
run_pruning_finetune "resnet50_softpool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_softpool_normal_cifar10_1784693143.032153_rf_level_4_recording_diverse_pooling_100_no_ffcv_test_acc_89.8.pth"
run_pruning_finetune "resnet50_softpool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_softpool_normal_cifar10_1784693448.2460418_rf_level_4_recording_diverse_pooling_100_no_ffcv_test_acc_89.41.pth"

# resnet50_mixedpool
run_pruning_finetune "resnet50_mixedpool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_mixedpool_normal_cifar10_1784666063.4725385_rf_level_1_recording_diverse_pooling_100_no_ffcv_test_acc_94.44.pth"
run_pruning_finetune "resnet50_mixedpool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_mixedpool_normal_cifar10_1784666109.52596_rf_level_1_recording_diverse_pooling_100_no_ffcv_test_acc_94.28.pth"
run_pruning_finetune "resnet50_mixedpool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_mixedpool_normal_cifar10_1784666256.8978019_rf_level_1_recording_diverse_pooling_100_no_ffcv_test_acc_94.16.pth"
run_pruning_finetune "resnet50_mixedpool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 2 "${resolution}" "${resize}" "resnet50_mixedpool_normal_cifar10_1784673889.4617448_rf_level_2_recording_diverse_pooling_100_no_ffcv_test_acc_93.3.pth"
run_pruning_finetune "resnet50_mixedpool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 2 "${resolution}" "${resize}" "resnet50_mixedpool_normal_cifar10_1784674012.354666_rf_level_2_recording_diverse_pooling_100_no_ffcv_test_acc_93.43.pth"
run_pruning_finetune "resnet50_mixedpool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 2 "${resolution}" "${resize}" "resnet50_mixedpool_normal_cifar10_1784679893.8735917_rf_level_2_recording_diverse_pooling_100_no_ffcv_test_acc_93.1.pth"
run_pruning_finetune "resnet50_mixedpool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 3 "${resolution}" "${resize}" "resnet50_mixedpool_normal_cifar10_1784687642.755106_rf_level_3_recording_diverse_pooling_100_no_ffcv_test_acc_91.83.pth"
run_pruning_finetune "resnet50_mixedpool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 3 "${resolution}" "${resize}" "resnet50_mixedpool_normal_cifar10_1784687643.075851_rf_level_3_recording_diverse_pooling_100_no_ffcv_test_acc_91.47.pth"
run_pruning_finetune "resnet50_mixedpool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 3 "${resolution}" "${resize}" "resnet50_mixedpool_normal_cifar10_1784687887.5424893_rf_level_3_recording_diverse_pooling_100_no_ffcv_test_acc_91.81.pth"
run_pruning_finetune "resnet50_mixedpool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_mixedpool_normal_cifar10_1784693447.1940536_rf_level_4_recording_diverse_pooling_100_no_ffcv_test_acc_89.68.pth"
run_pruning_finetune "resnet50_mixedpool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_mixedpool_normal_cifar10_1784693568.7674654_rf_level_4_recording_diverse_pooling_100_no_ffcv_test_acc_89.6.pth"
run_pruning_finetune "resnet50_mixedpool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_mixedpool_normal_cifar10_1784693568.7846038_rf_level_4_recording_diverse_pooling_100_no_ffcv_test_acc_89.5.pth"

# resnet50_lippool
run_pruning_finetune "resnet50_lippool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_lippool_normal_cifar10_1784671721.1267104_rf_level_1_recording_diverse_pooling_100_no_ffcv_test_acc_94.08.pth"
run_pruning_finetune "resnet50_lippool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_lippool_normal_cifar10_1784671783.9419346_rf_level_1_recording_diverse_pooling_100_no_ffcv_test_acc_94.58.pth"
run_pruning_finetune "resnet50_lippool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_lippool_normal_cifar10_1784671908.4666908_rf_level_1_recording_diverse_pooling_100_no_ffcv_test_acc_94.47.pth"
run_pruning_finetune "resnet50_lippool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 2 "${resolution}" "${resize}" "resnet50_lippool_normal_cifar10_1784680627.5685527_rf_level_2_recording_diverse_pooling_100_no_ffcv_test_acc_93.5.pth"
run_pruning_finetune "resnet50_lippool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 2 "${resolution}" "${resize}" "resnet50_lippool_normal_cifar10_1784680750.164549_rf_level_2_recording_diverse_pooling_100_no_ffcv_test_acc_93.31.pth"
run_pruning_finetune "resnet50_lippool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 2 "${resolution}" "${resize}" "resnet50_lippool_normal_cifar10_1784680933.0850174_rf_level_2_recording_diverse_pooling_100_no_ffcv_test_acc_94.05.pth"
run_pruning_finetune "resnet50_lippool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 3 "${resolution}" "${resize}" "resnet50_lippool_normal_cifar10_1784687948.5800734_rf_level_3_recording_diverse_pooling_100_no_ffcv_test_acc_92.13.pth"
run_pruning_finetune "resnet50_lippool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 3 "${resolution}" "${resize}" "resnet50_lippool_normal_cifar10_1784688070.5222235_rf_level_3_recording_diverse_pooling_100_no_ffcv_test_acc_92.5.pth"
run_pruning_finetune "resnet50_lippool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 3 "${resolution}" "${resize}" "resnet50_lippool_normal_cifar10_1784688254.4997098_rf_level_3_recording_diverse_pooling_100_no_ffcv_test_acc_92.33.pth"
run_pruning_finetune "resnet50_lippool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_lippool_normal_cifar10_1784693751.697758_rf_level_4_recording_diverse_pooling_100_no_ffcv_test_acc_90.48.pth"
run_pruning_finetune "resnet50_lippool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_lippool_normal_cifar10_1784693751.9207618_rf_level_4_recording_diverse_pooling_100_no_ffcv_test_acc_89.84.pth"
run_pruning_finetune "resnet50_lippool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_lippool_normal_cifar10_1784693812.727956_rf_level_4_recording_diverse_pooling_100_no_ffcv_test_acc_90.53.pth"

# vgg19_spectral_pool
run_pruning_finetune "vgg19_spectral_pool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_spectral_pool_normal_cifar10_1784631939.3204527_rf_level_1_recording_diverse_pooling_100_no_ffcv_test_acc_93.39.pth"
run_pruning_finetune "vgg19_spectral_pool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_spectral_pool_normal_cifar10_1784632109.2315927_rf_level_1_recording_diverse_pooling_100_no_ffcv_test_acc_93.67.pth"
run_pruning_finetune "vgg19_spectral_pool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_spectral_pool_normal_cifar10_1784634211.4273763_rf_level_1_recording_diverse_pooling_100_no_ffcv_test_acc_93.89.pth"
run_pruning_finetune "vgg19_spectral_pool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 2 "${resolution}" "${resize}" "vgg19_spectral_pool_normal_cifar10_1784637027.2295716_rf_level_2_recording_diverse_pooling_100_no_ffcv_test_acc_86.13.pth"
run_pruning_finetune "vgg19_spectral_pool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 2 "${resolution}" "${resize}" "vgg19_spectral_pool_normal_cifar10_1784637210.256399_rf_level_2_recording_diverse_pooling_100_no_ffcv_test_acc_87.03.pth"
run_pruning_finetune "vgg19_spectral_pool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 2 "${resolution}" "${resize}" "vgg19_spectral_pool_normal_cifar10_1784638563.5455806_rf_level_2_recording_diverse_pooling_100_no_ffcv_test_acc_86.45.pth"
run_pruning_finetune "vgg19_spectral_pool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 3 "${resolution}" "${resize}" "vgg19_spectral_pool_normal_cifar10_1784640883.5572007_rf_level_3_recording_diverse_pooling_100_no_ffcv_test_acc_82.02.pth"
run_pruning_finetune "vgg19_spectral_pool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 3 "${resolution}" "${resize}" "vgg19_spectral_pool_normal_cifar10_1784641913.507943_rf_level_3_recording_diverse_pooling_100_no_ffcv_test_acc_82.43.pth"
run_pruning_finetune "vgg19_spectral_pool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 3 "${resolution}" "${resize}" "vgg19_spectral_pool_normal_cifar10_1784641918.9527214_rf_level_3_recording_diverse_pooling_100_no_ffcv_test_acc_81.91.pth"
run_pruning_finetune "vgg19_spectral_pool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_spectral_pool_normal_cifar10_1784644794.5724225_rf_level_4_recording_diverse_pooling_100_no_ffcv_test_acc_71.14.pth"
run_pruning_finetune "vgg19_spectral_pool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_spectral_pool_normal_cifar10_1784644797.946559_rf_level_4_recording_diverse_pooling_100_no_ffcv_test_acc_73.03.pth"
run_pruning_finetune "vgg19_spectral_pool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_spectral_pool_normal_cifar10_1784644802.6702175_rf_level_4_recording_diverse_pooling_100_no_ffcv_test_acc_73.11.pth"

# vgg19_softpool
run_pruning_finetune "vgg19_softpool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_softpool_normal_cifar10_1784634335.4068856_rf_level_1_recording_diverse_pooling_100_no_ffcv_test_acc_92.52.pth"
run_pruning_finetune "vgg19_softpool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_softpool_normal_cifar10_1784634601.2182202_rf_level_1_recording_diverse_pooling_100_no_ffcv_test_acc_92.18.pth"
run_pruning_finetune "vgg19_softpool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_softpool_normal_cifar10_1784635161.6560163_rf_level_1_recording_diverse_pooling_100_no_ffcv_test_acc_92.61.pth"
run_pruning_finetune "vgg19_softpool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 2 "${resolution}" "${resize}" "vgg19_softpool_normal_cifar10_1784638629.9559612_rf_level_2_recording_diverse_pooling_100_no_ffcv_test_acc_89.03.pth"
run_pruning_finetune "vgg19_softpool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 2 "${resolution}" "${resize}" "vgg19_softpool_normal_cifar10_1784638779.0622687_rf_level_2_recording_diverse_pooling_100_no_ffcv_test_acc_89.36.pth"
run_pruning_finetune "vgg19_softpool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 2 "${resolution}" "${resize}" "vgg19_softpool_normal_cifar10_1784638789.2761524_rf_level_2_recording_diverse_pooling_100_no_ffcv_test_acc_89.29.pth"
run_pruning_finetune "vgg19_softpool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 3 "${resolution}" "${resize}" "vgg19_softpool_normal_cifar10_1784641937.390185_rf_level_3_recording_diverse_pooling_100_no_ffcv_test_acc_85.29.pth"
# SKIPPED (test_acc=10.0, looks like a failed/collapsed run): vgg19_softpool_normal_cifar10_1784642276.214949_rf_level_3_recording_diverse_pooling_100_no_ffcv_test_acc_10.0.pth
run_pruning_finetune "vgg19_softpool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 3 "${resolution}" "${resize}" "vgg19_softpool_normal_cifar10_1784642300.435114_rf_level_3_recording_diverse_pooling_100_no_ffcv_test_acc_85.73.pth"
run_pruning_finetune "vgg19_softpool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_softpool_normal_cifar10_1784644803.2006736_rf_level_4_recording_diverse_pooling_100_no_ffcv_test_acc_82.92.pth"
run_pruning_finetune "vgg19_softpool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_softpool_normal_cifar10_1784646666.810503_rf_level_4_recording_diverse_pooling_100_no_ffcv_test_acc_82.77.pth"
run_pruning_finetune "vgg19_softpool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_softpool_normal_cifar10_1784646666.8661034_rf_level_4_recording_diverse_pooling_100_no_ffcv_test_acc_78.38.pth"

# vgg19_mixedpool
run_pruning_finetune "vgg19_mixedpool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_mixedpool_normal_cifar10_1784635183.5906754_rf_level_1_recording_diverse_pooling_100_no_ffcv_test_acc_93.38.pth"
run_pruning_finetune "vgg19_mixedpool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_mixedpool_normal_cifar10_1784635458.1928606_rf_level_1_recording_diverse_pooling_100_no_ffcv_test_acc_93.53.pth"
run_pruning_finetune "vgg19_mixedpool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_mixedpool_normal_cifar10_1784636581.5009916_rf_level_1_recording_diverse_pooling_100_no_ffcv_test_acc_93.69.pth"
run_pruning_finetune "vgg19_mixedpool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 2 "${resolution}" "${resize}" "vgg19_mixedpool_normal_cifar10_1784638803.4968905_rf_level_2_recording_diverse_pooling_100_no_ffcv_test_acc_90.61.pth"
run_pruning_finetune "vgg19_mixedpool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 2 "${resolution}" "${resize}" "vgg19_mixedpool_normal_cifar10_1784638850.1895337_rf_level_2_recording_diverse_pooling_100_no_ffcv_test_acc_90.8.pth"
run_pruning_finetune "vgg19_mixedpool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 2 "${resolution}" "${resize}" "vgg19_mixedpool_normal_cifar10_1784639026.5070183_rf_level_2_recording_diverse_pooling_100_no_ffcv_test_acc_90.71.pth"
run_pruning_finetune "vgg19_mixedpool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 3 "${resolution}" "${resize}" "vgg19_mixedpool_normal_cifar10_1784642351.424408_rf_level_3_recording_diverse_pooling_100_no_ffcv_test_acc_88.0.pth"
run_pruning_finetune "vgg19_mixedpool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 3 "${resolution}" "${resize}" "vgg19_mixedpool_normal_cifar10_1784642413.6510293_rf_level_3_recording_diverse_pooling_100_no_ffcv_test_acc_88.46.pth"
run_pruning_finetune "vgg19_mixedpool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 3 "${resolution}" "${resize}" "vgg19_mixedpool_normal_cifar10_1784642571.4397159_rf_level_3_recording_diverse_pooling_100_no_ffcv_test_acc_88.36.pth"
run_pruning_finetune "vgg19_mixedpool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_mixedpool_normal_cifar10_1784646668.665527_rf_level_4_recording_diverse_pooling_100_no_ffcv_test_acc_85.35.pth"
run_pruning_finetune "vgg19_mixedpool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_mixedpool_normal_cifar10_1784647059.8361251_rf_level_4_recording_diverse_pooling_100_no_ffcv_test_acc_86.22.pth"
run_pruning_finetune "vgg19_mixedpool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_mixedpool_normal_cifar10_1784647309.6110876_rf_level_4_recording_diverse_pooling_100_no_ffcv_test_acc_81.47.pth"

# vgg19_lippool
run_pruning_finetune "vgg19_lippool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_lippool_normal_cifar10_1784636630.2052417_rf_level_1_recording_diverse_pooling_100_no_ffcv_test_acc_93.11.pth"
run_pruning_finetune "vgg19_lippool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_lippool_normal_cifar10_1784636746.0240648_rf_level_1_recording_diverse_pooling_100_no_ffcv_test_acc_93.06.pth"
run_pruning_finetune "vgg19_lippool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_lippool_normal_cifar10_1784636909.461994_rf_level_1_recording_diverse_pooling_100_no_ffcv_test_acc_93.1.pth"
run_pruning_finetune "vgg19_lippool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 2 "${resolution}" "${resize}" "vgg19_lippool_normal_cifar10_1784640546.0221145_rf_level_2_recording_diverse_pooling_100_no_ffcv_test_acc_90.7.pth"
run_pruning_finetune "vgg19_lippool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 2 "${resolution}" "${resize}" "vgg19_lippool_normal_cifar10_1784640550.8182352_rf_level_2_recording_diverse_pooling_100_no_ffcv_test_acc_90.47.pth"
run_pruning_finetune "vgg19_lippool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 2 "${resolution}" "${resize}" "vgg19_lippool_normal_cifar10_1784640599.0764744_rf_level_2_recording_diverse_pooling_100_no_ffcv_test_acc_90.87.pth"
run_pruning_finetune "vgg19_lippool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 3 "${resolution}" "${resize}" "vgg19_lippool_normal_cifar10_1784642899.5054936_rf_level_3_recording_diverse_pooling_100_no_ffcv_test_acc_87.63.pth"
run_pruning_finetune "vgg19_lippool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 3 "${resolution}" "${resize}" "vgg19_lippool_normal_cifar10_1784644791.2412736_rf_level_3_recording_diverse_pooling_100_no_ffcv_test_acc_88.12.pth"
run_pruning_finetune "vgg19_lippool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 3 "${resolution}" "${resize}" "vgg19_lippool_normal_cifar10_1784644791.6420498_rf_level_3_recording_diverse_pooling_100_no_ffcv_test_acc_87.75.pth"
run_pruning_finetune "vgg19_lippool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_lippool_normal_cifar10_1784647310.651843_rf_level_4_recording_diverse_pooling_100_no_ffcv_test_acc_86.74.pth"
run_pruning_finetune "vgg19_lippool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_lippool_normal_cifar10_1784647312.333187_rf_level_4_recording_diverse_pooling_100_no_ffcv_test_acc_83.06.pth"
run_pruning_finetune "vgg19_lippool" "${dataset}" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "recording_diverse_pooling_100_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_lippool_normal_cifar10_1784647328.2054644_rf_level_4_recording_diverse_pooling_100_no_ffcv_test_acc_83.96.pth"

done
