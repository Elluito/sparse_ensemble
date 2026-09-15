#!/bin/bash

# Prune + fine-tune (10 epochs) handler for the learning-rate-sweep checkpoints
# (vgg19, resnet50 on cifar10, rf_levels 1 and 4, all 7 learning rates from the
# sweep), mirroring slurm_diverse_pooling_pruning_handler.sh's run_pruning_finetune.
#
# Checkpoints live in "${SCRATCH}/learning_rate_experiments"
# (= /mnt/scratch/pznz720/learning_rate_experiments on this cluster).
# Each named SOLUTION is pruned at pruning_rate=0.9 (prune_models.py --experiment 2)
# and then fine-tuned for 10 epochs (hardcoded in slurm_learning_rate_pruning_finetune_run.sh).
# Output (fine-tuned checkpoint + accuracy log) lands under
# "${checkpoints_folder}/pruned/0.9" (prune_models.py's own convention, same as
# the diverse-pooling case) -- SAVE_FOLDER is passed through but unused by this
# experiment path.
#
# This covers every one of the 88 dense checkpoints produced by the lr sweep.

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
sbatch --nodes=1 --time=03:00:00 --partition=gpu --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="${model}_${rf_level}_${dataset}_${pruning_rate}_finetune_ffcv.err" --output="${model}_${rf_level}_${dataset}_${pruning_rate}_finetune_ffcv.out" --job-name="${model}_${rf_level}_${dataset}_${pruning_rate}_finetune_ffcv" slurm_learning_rate_pruning_finetune_run.sh FFCV="${ffcv}" NAME="${name}" MODEL="${model}" DATASET="${dataset}" NUMW=8 RFL="${rf_level}" TYPE="normal" FOLDER="${directory}" PR="${pruning_rate}" SOLUTION="${solution}" FFCV_TRAIN="${ffcv_train}" FFCV_VAL="${ffcv_val}" DATA_FOLDER="${data_folder}" SAVE_FOLDER="${save_folder}" INPUT_RES="${resolution}" RESIZE="${resize}"

else
 echo "Don't use FFCV"
 sbatch --nodes=1 --time=03:00:00 --partition=gpu --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="${model}_${rf_level}_${dataset}_${pruning_rate}_${name}_finetune.err" --output="${model}_${rf_level}_${dataset}_${pruning_rate}_${name}_finetune.out" --job-name="${model}_${rf_level}_${dataset}_${pruning_rate}_${name}_finetune" slurm_learning_rate_pruning_finetune_run.sh FFCV="${ffcv}" NAME="${name}" MODEL="${model}" DATASET="${dataset}" NUMW=8 RFL="${rf_level}" TYPE="normal" FOLDER="${directory}" PR="${pruning_rate}" SOLUTION="${solution}" DATA_FOLDER="${data_folder}" SAVE_FOLDER="${save_folder}" INPUT_RES="${resolution}" RESIZE="${resize}"
  fi
}

resolution=32
resize=0
pruning_rate=0.9
checkpoints_folder="${SCRATCH}/learning_rate_experiments"
save_folder="${HOME}/sparse_ensemble/learning_rate_sweep_pruning_finetune_results"


# resnet50 rf_level=1 lr=0.0001
# run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.0001_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788721931.1635652_rf_level_1_lr_0.0001_recording_lr_sweep_no_ffcv_test_acc_76.62.pth"
# run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.0001_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788722146.3243396_rf_level_1_lr_0.0001_recording_lr_sweep_no_ffcv_test_acc_77.02.pth"
# run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.0001_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788722237.4414394_rf_level_1_lr_0.0001_recording_lr_sweep_no_ffcv_test_acc_76.74.pth"

# resnet50 rf_level=1 lr=0.001
# run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.001_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788722294.128653_rf_level_1_lr_0.001_recording_lr_sweep_no_ffcv_test_acc_89.95.pth"
# run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.001_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788722460.444315_rf_level_1_lr_0.001_recording_lr_sweep_no_ffcv_test_acc_89.23.pth"
# run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.001_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788722524.8563814_rf_level_1_lr_0.001_recording_lr_sweep_no_ffcv_test_acc_89.16.pth"

# resnet50 rf_level=1 lr=0.003
# run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.003_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788722771.9463074_rf_level_1_lr_0.003_recording_lr_sweep_no_ffcv_test_acc_92.1.pth"
# run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.003_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788722780.9483254_rf_level_1_lr_0.003_recording_lr_sweep_no_ffcv_test_acc_92.27.pth"
# run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.003_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788723029.7261784_rf_level_1_lr_0.003_recording_lr_sweep_no_ffcv_test_acc_92.07.pth"

# resnet50 rf_level=1 lr=0.005
# run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.005_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788723040.2447066_rf_level_1_lr_0.005_recording_lr_sweep_no_ffcv_test_acc_93.06.pth"
# run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.005_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788723239.0105288_rf_level_1_lr_0.005_recording_lr_sweep_no_ffcv_test_acc_92.85.pth"
# run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.005_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788723250.185689_rf_level_1_lr_0.005_recording_lr_sweep_no_ffcv_test_acc_92.86.pth"

# resnet50 rf_level=1 lr=0.007
# run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.007_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788723570.045012_rf_level_1_lr_0.007_recording_lr_sweep_no_ffcv_test_acc_93.39.pth"
# run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.007_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788723632.0164754_rf_level_1_lr_0.007_recording_lr_sweep_no_ffcv_test_acc_93.52.pth"
# run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.007_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788724893.9668913_rf_level_1_lr_0.007_recording_lr_sweep_no_ffcv_test_acc_93.48.pth"

# resnet50 rf_level=1 lr=0.1
# run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.1_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788743892.6023967_rf_level_1_lr_0.1_recording_lr_sweep_no_ffcv_test_acc_94.29.pth"
# run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.1_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788744031.3381374_rf_level_1_lr_0.1_recording_lr_sweep_no_ffcv_test_acc_94.98.pth"
# run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.1_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788744344.4921055_rf_level_1_lr_0.1_recording_lr_sweep_no_ffcv_test_acc_94.6.pth"

# resnet50 rf_level=1 lr=0.2
# run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.2_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788744446.6563656_rf_level_1_lr_0.2_recording_lr_sweep_no_ffcv_test_acc_94.07.pth"
# run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.2_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788744744.4855676_rf_level_1_lr_0.2_recording_lr_sweep_no_ffcv_test_acc_94.04.pth"
# run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.2_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788744838.3064055_rf_level_1_lr_0.2_recording_lr_sweep_no_ffcv_test_acc_93.91.pth"

# resnet50 rf_level=4 lr=0.0001
# run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.0001_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788744840.882203_rf_level_4_lr_0.0001_recording_lr_sweep_no_ffcv_test_acc_69.64.pth"
# run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.0001_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788744898.7421641_rf_level_4_lr_0.0001_recording_lr_sweep_no_ffcv_test_acc_70.43.pth"
# run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.0001_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788745037.2133198_rf_level_4_lr_0.0001_recording_lr_sweep_no_ffcv_test_acc_68.4.pth"

# resnet50 rf_level=4 lr=0.001
# run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.001_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788745401.9234877_rf_level_4_lr_0.001_recording_lr_sweep_no_ffcv_test_acc_82.12.pth"
# run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.001_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788746722.6990306_rf_level_4_lr_0.001_recording_lr_sweep_no_ffcv_test_acc_82.45.pth"
# run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.001_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788747010.1366668_rf_level_4_lr_0.001_recording_lr_sweep_no_ffcv_test_acc_81.99.pth"

# resnet50 rf_level=4 lr=0.003
# run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.003_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788747796.701767_rf_level_4_lr_0.003_recording_lr_sweep_no_ffcv_test_acc_84.4.pth"
# run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.003_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788748038.89654_rf_level_4_lr_0.003_recording_lr_sweep_no_ffcv_test_acc_83.4.pth"
# run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.003_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788748593.7011423_rf_level_4_lr_0.003_recording_lr_sweep_no_ffcv_test_acc_83.74.pth"

# resnet50 rf_level=4 lr=0.005
# run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.005_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788749033.7146225_rf_level_4_lr_0.005_recording_lr_sweep_no_ffcv_test_acc_84.1.pth"
# run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.005_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788749539.8593116_rf_level_4_lr_0.005_recording_lr_sweep_no_ffcv_test_acc_81.18.pth"
# run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.005_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788749848.3423297_rf_level_4_lr_0.005_recording_lr_sweep_no_ffcv_test_acc_77.75.pth"

# resnet50 rf_level=4 lr=0.007
# run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.007_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788751625.7091773_rf_level_4_lr_0.007_recording_lr_sweep_no_ffcv_test_acc_83.13.pth"
# run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.007_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788751844.2021394_rf_level_4_lr_0.007_recording_lr_sweep_no_ffcv_test_acc_83.86.pth"
# run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.007_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788751860.8808124_rf_level_4_lr_0.007_recording_lr_sweep_no_ffcv_test_acc_84.23.pth"

# resnet50 rf_level=4 lr=0.1
# run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.1_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788752624.71632_rf_level_4_lr_0.1_recording_lr_sweep_no_ffcv_test_acc_89.7.pth"
# run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.1_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788752830.3645015_rf_level_4_lr_0.1_recording_lr_sweep_no_ffcv_test_acc_89.5.pth"
# run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.1_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788753283.4540498_rf_level_4_lr_0.1_recording_lr_sweep_no_ffcv_test_acc_89.73.pth"

# resnet50 rf_level=4 lr=0.2
# run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.2_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788754014.7215848_rf_level_4_lr_0.2_recording_lr_sweep_no_ffcv_test_acc_89.51.pth"
# run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.2_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788754115.6596978_rf_level_4_lr_0.2_recording_lr_sweep_no_ffcv_test_acc_89.08.pth"
# run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.2_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788754589.922838_rf_level_4_lr_0.2_recording_lr_sweep_no_ffcv_test_acc_89.32.pth"

# vgg19 rf_level=1 lr=0.0001
# run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.0001_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788712914.2861767_rf_level_1_lr_0.0001_recording_lr_sweep_no_ffcv_test_acc_82.56.pth"
# run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.0001_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788713022.946601_rf_level_1_lr_0.0001_recording_lr_sweep_no_ffcv_test_acc_80.82.pth"
# run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.0001_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788713117.0697205_rf_level_1_lr_0.0001_recording_lr_sweep_no_ffcv_test_acc_81.53.pth"
# run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.0001_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788715485.5843997_rf_level_1_lr_0.0001_recording_lr_sweep_no_ffcv_test_acc_82.87.pth"
# run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.0001_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788715503.2196443_rf_level_1_lr_0.0001_recording_lr_sweep_no_ffcv_test_acc_82.4.pth"
# run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.0001_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788715532.770455_rf_level_1_lr_0.0001_recording_lr_sweep_no_ffcv_test_acc_82.28.pth"

# vgg19 rf_level=1 lr=0.001
# run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.001_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788713209.7757642_rf_level_1_lr_0.001_recording_lr_sweep_no_ffcv_test_acc_88.04.pth"
# run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.001_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788715610.0865316_rf_level_1_lr_0.001_recording_lr_sweep_no_ffcv_test_acc_89.58.pth"
# run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.001_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788715730.759108_rf_level_1_lr_0.001_recording_lr_sweep_no_ffcv_test_acc_90.08.pth"
# run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.001_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788715800.7998536_rf_level_1_lr_0.001_recording_lr_sweep_no_ffcv_test_acc_90.1.pth"

# vgg19 rf_level=1 lr=0.003
# run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.003_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788715995.7084787_rf_level_1_lr_0.003_recording_lr_sweep_no_ffcv_test_acc_91.87.pth"
# run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.003_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788716748.5713255_rf_level_1_lr_0.003_recording_lr_sweep_no_ffcv_test_acc_91.91.pth"
# run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.003_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788716776.1972272_rf_level_1_lr_0.003_recording_lr_sweep_no_ffcv_test_acc_91.42.pth"

# vgg19 rf_level=1 lr=0.005
# run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.005_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788716861.0371158_rf_level_1_lr_0.005_recording_lr_sweep_no_ffcv_test_acc_92.27.pth"
# run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.005_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788717482.3884494_rf_level_1_lr_0.005_recording_lr_sweep_no_ffcv_test_acc_92.14.pth"
# run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.005_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788717492.499358_rf_level_1_lr_0.005_recording_lr_sweep_no_ffcv_test_acc_92.22.pth"

# vgg19 rf_level=1 lr=0.007
# run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.007_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788717512.386441_rf_level_1_lr_0.007_recording_lr_sweep_no_ffcv_test_acc_92.1.pth"
# run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.007_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788717540.3817_rf_level_1_lr_0.007_recording_lr_sweep_no_ffcv_test_acc_92.63.pth"
# run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.007_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788717635.3857872_rf_level_1_lr_0.007_recording_lr_sweep_no_ffcv_test_acc_92.47.pth"

# vgg19 rf_level=1 lr=0.1
# run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.1_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788717712.5394752_rf_level_1_lr_0.1_recording_lr_sweep_no_ffcv_test_acc_93.07.pth"
# run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.1_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788717731.6579177_rf_level_1_lr_0.1_recording_lr_sweep_no_ffcv_test_acc_92.9.pth"
# run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.1_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788717962.1773107_rf_level_1_lr_0.1_recording_lr_sweep_no_ffcv_test_acc_92.95.pth"

# vgg19 rf_level=1 lr=0.2
# run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.2_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788718163.7206402_rf_level_1_lr_0.2_recording_lr_sweep_no_ffcv_test_acc_92.97.pth"
# run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.2_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788718205.4628863_rf_level_1_lr_0.2_recording_lr_sweep_no_ffcv_test_acc_93.13.pth"
# run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.2_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788718632.2567356_rf_level_1_lr_0.2_recording_lr_sweep_no_ffcv_test_acc_92.9.pth"

# vgg19 rf_level=4 lr=0.0001
# run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.0001_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788718772.9758093_rf_level_4_lr_0.0001_recording_lr_sweep_no_ffcv_test_acc_71.18.pth"
# run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.0001_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788718785.3818936_rf_level_4_lr_0.0001_recording_lr_sweep_no_ffcv_test_acc_71.13.pth"
# run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.0001_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788718787.6567783_rf_level_4_lr_0.0001_recording_lr_sweep_no_ffcv_test_acc_71.11.pth"

# vgg19 rf_level=4 lr=0.001
# run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.001_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788719410.719972_rf_level_4_lr_0.001_recording_lr_sweep_no_ffcv_test_acc_80.89.pth"
# run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.001_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788719415.1342764_rf_level_4_lr_0.001_recording_lr_sweep_no_ffcv_test_acc_80.35.pth"
# run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.001_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788719438.0126386_rf_level_4_lr_0.001_recording_lr_sweep_no_ffcv_test_acc_80.47.pth"

# vgg19 rf_level=4 lr=0.003
# run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.003_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788719479.7494066_rf_level_4_lr_0.003_recording_lr_sweep_no_ffcv_test_acc_83.66.pth"
# run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.003_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788719719.0861125_rf_level_4_lr_0.003_recording_lr_sweep_no_ffcv_test_acc_83.95.pth"
# run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.003_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788719733.595991_rf_level_4_lr_0.003_recording_lr_sweep_no_ffcv_test_acc_84.03.pth"

# vgg19 rf_level=4 lr=0.005
# run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.005_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788719963.1658373_rf_level_4_lr_0.005_recording_lr_sweep_no_ffcv_test_acc_83.8.pth"
# run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.005_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788720116.3774176_rf_level_4_lr_0.005_recording_lr_sweep_no_ffcv_test_acc_84.66.pth"
# run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.005_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788720368.181866_rf_level_4_lr_0.005_recording_lr_sweep_no_ffcv_test_acc_84.14.pth"

# vgg19 rf_level=4 lr=0.007
# run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.007_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788720478.6662486_rf_level_4_lr_0.007_recording_lr_sweep_no_ffcv_test_acc_84.78.pth"
# run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.007_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788720601.4875767_rf_level_4_lr_0.007_recording_lr_sweep_no_ffcv_test_acc_84.44.pth"
# run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.007_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788720704.6770127_rf_level_4_lr_0.007_recording_lr_sweep_no_ffcv_test_acc_84.86.pth"

# vgg19 rf_level=4 lr=0.1
# run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.1_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788720834.7581613_rf_level_4_lr_0.1_recording_lr_sweep_no_ffcv_test_acc_85.46.pth"
# run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.1_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788721215.3549552_rf_level_4_lr_0.1_recording_lr_sweep_no_ffcv_test_acc_85.04.pth"
# run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.1_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788721245.219036_rf_level_4_lr_0.1_recording_lr_sweep_no_ffcv_test_acc_84.95.pth"

# vgg19 rf_level=4 lr=0.2
# run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.2_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788721326.9660506_rf_level_4_lr_0.2_recording_lr_sweep_no_ffcv_test_acc_79.37.pth"
# run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.2_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788721355.7628684_rf_level_4_lr_0.2_recording_lr_sweep_no_ffcv_test_acc_84.88.pth"
# run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.2_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788721706.5880888_rf_level_4_lr_0.2_recording_lr_sweep_no_ffcv_test_acc_79.88.pth"

#############################################################
# Additional learning rates: 0.01 0.02 0.03 0.04 0.05 0.06 0.07
# 0.08 0.09 0.15 (same models/rf_levels/pruning_rate as above).
#############################################################
# resnet50 rf_level=1 lr=0.01
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.01_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789183560.5133927_rf_level_1_lr_0.01_recording_lr_sweep_no_ffcv_test_acc_94.12.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.01_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789183685.6168153_rf_level_1_lr_0.01_recording_lr_sweep_no_ffcv_test_acc_93.79.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.01_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789183860.4198062_rf_level_1_lr_0.01_recording_lr_sweep_no_ffcv_test_acc_93.83.pth"

# resnet50 rf_level=1 lr=0.02
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.02_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789160832.8743768_rf_level_1_lr_0.02_recording_lr_sweep_no_ffcv_test_acc_94.02.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.02_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789160833.610254_rf_level_1_lr_0.02_recording_lr_sweep_no_ffcv_test_acc_94.06.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.02_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789160833.7196825_rf_level_1_lr_0.02_recording_lr_sweep_no_ffcv_test_acc_94.48.pth"

# resnet50 rf_level=1 lr=0.03
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.03_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789160835.6978335_rf_level_1_lr_0.03_recording_lr_sweep_no_ffcv_test_acc_94.06.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.03_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789160836.295753_rf_level_1_lr_0.03_recording_lr_sweep_no_ffcv_test_acc_93.95.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.03_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789161105.2254102_rf_level_1_lr_0.03_recording_lr_sweep_no_ffcv_test_acc_94.38.pth"

# resnet50 rf_level=1 lr=0.04
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.04_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789161181.4498477_rf_level_1_lr_0.04_recording_lr_sweep_no_ffcv_test_acc_94.43.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.04_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789161341.3143895_rf_level_1_lr_0.04_recording_lr_sweep_no_ffcv_test_acc_94.47.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.04_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789161729.6134002_rf_level_1_lr_0.04_recording_lr_sweep_no_ffcv_test_acc_93.97.pth"

# resnet50 rf_level=1 lr=0.05
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.05_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789161792.1362278_rf_level_1_lr_0.05_recording_lr_sweep_no_ffcv_test_acc_94.59.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.05_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789161846.771495_rf_level_1_lr_0.05_recording_lr_sweep_no_ffcv_test_acc_94.64.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.05_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789162025.5796146_rf_level_1_lr_0.05_recording_lr_sweep_no_ffcv_test_acc_94.44.pth"

# resnet50 rf_level=1 lr=0.06
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.06_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789162049.6430233_rf_level_1_lr_0.06_recording_lr_sweep_no_ffcv_test_acc_94.62.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.06_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789162166.7431448_rf_level_1_lr_0.06_recording_lr_sweep_no_ffcv_test_acc_94.68.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.06_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789163154.1470792_rf_level_1_lr_0.06_recording_lr_sweep_no_ffcv_test_acc_94.58.pth"

# resnet50 rf_level=1 lr=0.07
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.07_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789182543.688598_rf_level_1_lr_0.07_recording_lr_sweep_no_ffcv_test_acc_95.03.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.07_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789182543.688615_rf_level_1_lr_0.07_recording_lr_sweep_no_ffcv_test_acc_94.54.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.07_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789182724.963365_rf_level_1_lr_0.07_recording_lr_sweep_no_ffcv_test_acc_94.85.pth"

# resnet50 rf_level=1 lr=0.08
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.08_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789182725.0992155_rf_level_1_lr_0.08_recording_lr_sweep_no_ffcv_test_acc_94.7.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.08_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789183144.8918927_rf_level_1_lr_0.08_recording_lr_sweep_no_ffcv_test_acc_94.28.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.08_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789183205.3389533_rf_level_1_lr_0.08_recording_lr_sweep_no_ffcv_test_acc_94.46.pth"

# resnet50 rf_level=1 lr=0.09
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.09_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789183325.5402431_rf_level_1_lr_0.09_recording_lr_sweep_no_ffcv_test_acc_94.39.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.09_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789183439.6791983_rf_level_1_lr_0.09_recording_lr_sweep_no_ffcv_test_acc_93.82.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.09_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789183445.2731376_rf_level_1_lr_0.09_recording_lr_sweep_no_ffcv_test_acc_94.17.pth"

# resnet50 rf_level=1 lr=0.15
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.15_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789184046.5863907_rf_level_1_lr_0.15_recording_lr_sweep_no_ffcv_test_acc_94.36.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.15_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789184581.6250136_rf_level_1_lr_0.15_recording_lr_sweep_no_ffcv_test_acc_94.42.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.15_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789184821.8853803_rf_level_1_lr_0.15_recording_lr_sweep_no_ffcv_test_acc_94.44.pth"

# resnet50 rf_level=4 lr=0.01
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.01_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789236090.4546285_rf_level_4_lr_0.01_recording_lr_sweep_no_ffcv_test_acc_80.34.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.01_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789236090.4880545_rf_level_4_lr_0.01_recording_lr_sweep_no_ffcv_test_acc_84.48.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.01_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789236094.9747922_rf_level_4_lr_0.01_recording_lr_sweep_no_ffcv_test_acc_84.75.pth"

# resnet50 rf_level=4 lr=0.02
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.02_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789204358.1842608_rf_level_4_lr_0.02_recording_lr_sweep_no_ffcv_test_acc_88.09.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.02_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789204358.1843143_rf_level_4_lr_0.02_recording_lr_sweep_no_ffcv_test_acc_87.14.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.02_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789204590.1810987_rf_level_4_lr_0.02_recording_lr_sweep_no_ffcv_test_acc_86.65.pth"

# resnet50 rf_level=4 lr=0.03
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.03_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789205123.6333714_rf_level_4_lr_0.03_recording_lr_sweep_no_ffcv_test_acc_88.77.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.03_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789205123.6856034_rf_level_4_lr_0.03_recording_lr_sweep_no_ffcv_test_acc_88.69.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.03_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789205202.5794132_rf_level_4_lr_0.03_recording_lr_sweep_no_ffcv_test_acc_88.38.pth"

# resnet50 rf_level=4 lr=0.04
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.04_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789205243.8333123_rf_level_4_lr_0.04_recording_lr_sweep_no_ffcv_test_acc_88.68.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.04_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789205249.2420537_rf_level_4_lr_0.04_recording_lr_sweep_no_ffcv_test_acc_89.51.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.04_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789205429.4567637_rf_level_4_lr_0.04_recording_lr_sweep_no_ffcv_test_acc_90.09.pth"

# resnet50 rf_level=4 lr=0.05
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.05_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789205854.9766822_rf_level_4_lr_0.05_recording_lr_sweep_no_ffcv_test_acc_89.64.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.05_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789205910.3385916_rf_level_4_lr_0.05_recording_lr_sweep_no_ffcv_test_acc_89.7.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.05_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789206149.9157653_rf_level_4_lr_0.05_recording_lr_sweep_no_ffcv_test_acc_86.63.pth"

# resnet50 rf_level=4 lr=0.06
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.06_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789206330.1272056_rf_level_4_lr_0.06_recording_lr_sweep_no_ffcv_test_acc_89.92.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.06_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789206384.3675017_rf_level_4_lr_0.06_recording_lr_sweep_no_ffcv_test_acc_89.21.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.06_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789206564.402291_rf_level_4_lr_0.06_recording_lr_sweep_no_ffcv_test_acc_89.28.pth"

# resnet50 rf_level=4 lr=0.07
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.07_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789235189.1367218_rf_level_4_lr_0.07_recording_lr_sweep_no_ffcv_test_acc_88.75.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.07_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789235189.136726_rf_level_4_lr_0.07_recording_lr_sweep_no_ffcv_test_acc_90.17.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.07_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789235189.1367705_rf_level_4_lr_0.07_recording_lr_sweep_no_ffcv_test_acc_89.49.pth"

# resnet50 rf_level=4 lr=0.08
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.08_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789235637.5667057_rf_level_4_lr_0.08_recording_lr_sweep_no_ffcv_test_acc_90.06.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.08_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789235659.1183896_rf_level_4_lr_0.08_recording_lr_sweep_no_ffcv_test_acc_89.34.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.08_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789235777.829325_rf_level_4_lr_0.08_recording_lr_sweep_no_ffcv_test_acc_89.85.pth"

# resnet50 rf_level=4 lr=0.09
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.09_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789235851.61043_rf_level_4_lr_0.09_recording_lr_sweep_no_ffcv_test_acc_89.33.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.09_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789235924.6308892_rf_level_4_lr_0.09_recording_lr_sweep_no_ffcv_test_acc_88.71.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.09_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789235995.2198417_rf_level_4_lr_0.09_recording_lr_sweep_no_ffcv_test_acc_89.63.pth"

# resnet50 rf_level=4 lr=0.15
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.15_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789236090.4232898_rf_level_4_lr_0.15_recording_lr_sweep_no_ffcv_test_acc_89.81.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.15_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789236090.7197547_rf_level_4_lr_0.15_recording_lr_sweep_no_ffcv_test_acc_88.92.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.15_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1789236120.2571008_rf_level_4_lr_0.15_recording_lr_sweep_no_ffcv_test_acc_89.62.pth"

# vgg19 rf_level=1 lr=0.01
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.01_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789155100.4237812_rf_level_1_lr_0.01_recording_lr_sweep_no_ffcv_test_acc_92.48.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.01_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789155168.9885755_rf_level_1_lr_0.01_recording_lr_sweep_no_ffcv_test_acc_92.49.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.01_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789155173.1845274_rf_level_1_lr_0.01_recording_lr_sweep_no_ffcv_test_acc_92.73.pth"

# vgg19 rf_level=1 lr=0.02
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.02_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789151238.5416589_rf_level_1_lr_0.02_recording_lr_sweep_no_ffcv_test_acc_92.78.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.02_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789151268.7593865_rf_level_1_lr_0.02_recording_lr_sweep_no_ffcv_test_acc_92.81.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.02_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789151381.9131823_rf_level_1_lr_0.02_recording_lr_sweep_no_ffcv_test_acc_92.89.pth"

# vgg19 rf_level=1 lr=0.03
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.03_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789151375.918908_rf_level_1_lr_0.03_recording_lr_sweep_no_ffcv_test_acc_92.85.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.03_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789151381.7698495_rf_level_1_lr_0.03_recording_lr_sweep_no_ffcv_test_acc_92.82.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.03_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789151382.1595109_rf_level_1_lr_0.03_recording_lr_sweep_no_ffcv_test_acc_93.13.pth"

# vgg19 rf_level=1 lr=0.04
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.04_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789151381.926881_rf_level_1_lr_0.04_recording_lr_sweep_no_ffcv_test_acc_93.13.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.04_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789151381.9327056_rf_level_1_lr_0.04_recording_lr_sweep_no_ffcv_test_acc_92.8.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.04_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789151382.2231526_rf_level_1_lr_0.04_recording_lr_sweep_no_ffcv_test_acc_93.0.pth"

# vgg19 rf_level=1 lr=0.05
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.05_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789153201.390625_rf_level_1_lr_0.05_recording_lr_sweep_no_ffcv_test_acc_93.0.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.05_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789153205.0396035_rf_level_1_lr_0.05_recording_lr_sweep_no_ffcv_test_acc_93.01.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.05_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789153205.7355864_rf_level_1_lr_0.05_recording_lr_sweep_no_ffcv_test_acc_93.02.pth"

# vgg19 rf_level=1 lr=0.06
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.06_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789153254.9166515_rf_level_1_lr_0.06_recording_lr_sweep_no_ffcv_test_acc_93.18.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.06_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789153299.327594_rf_level_1_lr_0.06_recording_lr_sweep_no_ffcv_test_acc_93.31.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.06_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789153390.3112817_rf_level_1_lr_0.06_recording_lr_sweep_no_ffcv_test_acc_93.02.pth"

# vgg19 rf_level=1 lr=0.07
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.07_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789153450.9556742_rf_level_1_lr_0.07_recording_lr_sweep_no_ffcv_test_acc_93.13.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.07_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789153599.7239406_rf_level_1_lr_0.07_recording_lr_sweep_no_ffcv_test_acc_93.18.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.07_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789153811.448231_rf_level_1_lr_0.07_recording_lr_sweep_no_ffcv_test_acc_92.96.pth"

# vgg19 rf_level=1 lr=0.08
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.08_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789153980.3500066_rf_level_1_lr_0.08_recording_lr_sweep_no_ffcv_test_acc_93.35.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.08_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789154149.993401_rf_level_1_lr_0.08_recording_lr_sweep_no_ffcv_test_acc_93.1.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.08_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789154853.2772586_rf_level_1_lr_0.08_recording_lr_sweep_no_ffcv_test_acc_93.26.pth"

# vgg19 rf_level=1 lr=0.09
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.09_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789154977.0959504_rf_level_1_lr_0.09_recording_lr_sweep_no_ffcv_test_acc_93.43.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.09_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789154979.0766444_rf_level_1_lr_0.09_recording_lr_sweep_no_ffcv_test_acc_93.28.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.09_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789155072.5062509_rf_level_1_lr_0.09_recording_lr_sweep_no_ffcv_test_acc_93.21.pth"

# vgg19 rf_level=1 lr=0.15
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.15_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789155244.9528005_rf_level_1_lr_0.15_recording_lr_sweep_no_ffcv_test_acc_93.3.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.15_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789155477.4103026_rf_level_1_lr_0.15_recording_lr_sweep_no_ffcv_test_acc_93.27.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.15_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789155500.7451427_rf_level_1_lr_0.15_recording_lr_sweep_no_ffcv_test_acc_93.23.pth"

# vgg19 rf_level=4 lr=0.01
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.01_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789159192.3772376_rf_level_4_lr_0.01_recording_lr_sweep_no_ffcv_test_acc_85.32.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.01_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789159192.4504473_rf_level_4_lr_0.01_recording_lr_sweep_no_ffcv_test_acc_84.88.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.01_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789159372.5895252_rf_level_4_lr_0.01_recording_lr_sweep_no_ffcv_test_acc_84.92.pth"

# vgg19 rf_level=4 lr=0.02
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.02_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789155524.3949537_rf_level_4_lr_0.02_recording_lr_sweep_no_ffcv_test_acc_85.11.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.02_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789155873.146939_rf_level_4_lr_0.02_recording_lr_sweep_no_ffcv_test_acc_85.23.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.02_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789156053.706918_rf_level_4_lr_0.02_recording_lr_sweep_no_ffcv_test_acc_86.1.pth"

# vgg19 rf_level=4 lr=0.03
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.03_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789156212.5462513_rf_level_4_lr_0.03_recording_lr_sweep_no_ffcv_test_acc_85.31.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.03_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789156785.6477222_rf_level_4_lr_0.03_recording_lr_sweep_no_ffcv_test_acc_85.41.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.03_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789156851.7949462_rf_level_4_lr_0.03_recording_lr_sweep_no_ffcv_test_acc_85.29.pth"

# vgg19 rf_level=4 lr=0.04
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.04_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789156905.9440978_rf_level_4_lr_0.04_recording_lr_sweep_no_ffcv_test_acc_85.3.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.04_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789156967.0429962_rf_level_4_lr_0.04_recording_lr_sweep_no_ffcv_test_acc_85.68.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.04_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789156970.0695884_rf_level_4_lr_0.04_recording_lr_sweep_no_ffcv_test_acc_85.64.pth"

# vgg19 rf_level=4 lr=0.05
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.05_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789157026.2439182_rf_level_4_lr_0.05_recording_lr_sweep_no_ffcv_test_acc_85.98.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.05_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789157092.1062582_rf_level_4_lr_0.05_recording_lr_sweep_no_ffcv_test_acc_85.6.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.05_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789157147.008378_rf_level_4_lr_0.05_recording_lr_sweep_no_ffcv_test_acc_85.91.pth"

# vgg19 rf_level=4 lr=0.06
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.06_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789157387.248265_rf_level_4_lr_0.06_recording_lr_sweep_no_ffcv_test_acc_85.36.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.06_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789157689.0326927_rf_level_4_lr_0.06_recording_lr_sweep_no_ffcv_test_acc_85.5.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.06_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789158198.1901937_rf_level_4_lr_0.06_recording_lr_sweep_no_ffcv_test_acc_85.66.pth"

# vgg19 rf_level=4 lr=0.07
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.07_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789158230.8872607_rf_level_4_lr_0.07_recording_lr_sweep_no_ffcv_test_acc_84.97.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.07_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789158289.729932_rf_level_4_lr_0.07_recording_lr_sweep_no_ffcv_test_acc_85.61.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.07_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789158294.1853418_rf_level_4_lr_0.07_recording_lr_sweep_no_ffcv_test_acc_85.03.pth"

# vgg19 rf_level=4 lr=0.08
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.08_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789158594.3271492_rf_level_4_lr_0.08_recording_lr_sweep_no_ffcv_test_acc_85.51.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.08_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789158650.5087202_rf_level_4_lr_0.08_recording_lr_sweep_no_ffcv_test_acc_85.67.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.08_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789158711.1499183_rf_level_4_lr_0.08_recording_lr_sweep_no_ffcv_test_acc_85.85.pth"

# vgg19 rf_level=4 lr=0.09
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.09_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789158831.2676575_rf_level_4_lr_0.09_recording_lr_sweep_no_ffcv_test_acc_85.2.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.09_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789158952.964153_rf_level_4_lr_0.09_recording_lr_sweep_no_ffcv_test_acc_85.49.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.09_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789159012.9612083_rf_level_4_lr_0.09_recording_lr_sweep_no_ffcv_test_acc_85.38.pth"

# vgg19 rf_level=4 lr=0.15
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.15_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789159674.8744388_rf_level_4_lr_0.15_recording_lr_sweep_no_ffcv_test_acc_77.19.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.15_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789159854.6474957_rf_level_4_lr_0.15_recording_lr_sweep_no_ffcv_test_acc_85.09.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.15_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1789160275.0345554_rf_level_4_lr_0.15_recording_lr_sweep_no_ffcv_test_acc_72.82.pth"
