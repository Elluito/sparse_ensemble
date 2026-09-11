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
 sbatch --nodes=1 --time=03:00:00 --partition=gpu --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="${model}_${rf_level}_${dataset}_${pruning_rate}_finetune_no_ffcv.err" --output="${model}_${rf_level}_${dataset}_${pruning_rate}_finetune_no_ffcv.out" --job-name="${model}_${rf_level}_${dataset}_${pruning_rate}_finetune_no_ffcv" slurm_learning_rate_pruning_finetune_run.sh FFCV="${ffcv}" NAME="${name}" MODEL="${model}" DATASET="${dataset}" NUMW=8 RFL="${rf_level}" TYPE="normal" FOLDER="${directory}" PR="${pruning_rate}" SOLUTION="${solution}" DATA_FOLDER="${data_folder}" SAVE_FOLDER="${save_folder}" INPUT_RES="${resolution}" RESIZE="${resize}"
  fi
}

resolution=32
resize=0
pruning_rate=0.9
checkpoints_folder="${SCRATCH}/learning_rate_experiments"
save_folder="${HOME}/sparse_ensemble/learning_rate_sweep_pruning_finetune_results"


# resnet50 rf_level=1 lr=0.0001
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.0001_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788721931.1635652_rf_level_1_lr_0.0001_recording_lr_sweep_no_ffcv_test_acc_76.62.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.0001_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788722146.3243396_rf_level_1_lr_0.0001_recording_lr_sweep_no_ffcv_test_acc_77.02.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.0001_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788722237.4414394_rf_level_1_lr_0.0001_recording_lr_sweep_no_ffcv_test_acc_76.74.pth"

# resnet50 rf_level=1 lr=0.001
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.001_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788722294.128653_rf_level_1_lr_0.001_recording_lr_sweep_no_ffcv_test_acc_89.95.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.001_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788722460.444315_rf_level_1_lr_0.001_recording_lr_sweep_no_ffcv_test_acc_89.23.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.001_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788722524.8563814_rf_level_1_lr_0.001_recording_lr_sweep_no_ffcv_test_acc_89.16.pth"

# resnet50 rf_level=1 lr=0.003
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.003_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788722771.9463074_rf_level_1_lr_0.003_recording_lr_sweep_no_ffcv_test_acc_92.1.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.003_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788722780.9483254_rf_level_1_lr_0.003_recording_lr_sweep_no_ffcv_test_acc_92.27.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.003_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788723029.7261784_rf_level_1_lr_0.003_recording_lr_sweep_no_ffcv_test_acc_92.07.pth"

# resnet50 rf_level=1 lr=0.005
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.005_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788723040.2447066_rf_level_1_lr_0.005_recording_lr_sweep_no_ffcv_test_acc_93.06.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.005_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788723239.0105288_rf_level_1_lr_0.005_recording_lr_sweep_no_ffcv_test_acc_92.85.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.005_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788723250.185689_rf_level_1_lr_0.005_recording_lr_sweep_no_ffcv_test_acc_92.86.pth"

# resnet50 rf_level=1 lr=0.007
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.007_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788723570.045012_rf_level_1_lr_0.007_recording_lr_sweep_no_ffcv_test_acc_93.39.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.007_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788723632.0164754_rf_level_1_lr_0.007_recording_lr_sweep_no_ffcv_test_acc_93.52.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.007_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788724893.9668913_rf_level_1_lr_0.007_recording_lr_sweep_no_ffcv_test_acc_93.48.pth"

# resnet50 rf_level=1 lr=0.1
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.1_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788743892.6023967_rf_level_1_lr_0.1_recording_lr_sweep_no_ffcv_test_acc_94.29.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.1_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788744031.3381374_rf_level_1_lr_0.1_recording_lr_sweep_no_ffcv_test_acc_94.98.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.1_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788744344.4921055_rf_level_1_lr_0.1_recording_lr_sweep_no_ffcv_test_acc_94.6.pth"

# resnet50 rf_level=1 lr=0.2
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.2_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788744446.6563656_rf_level_1_lr_0.2_recording_lr_sweep_no_ffcv_test_acc_94.07.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.2_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788744744.4855676_rf_level_1_lr_0.2_recording_lr_sweep_no_ffcv_test_acc_94.04.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.2_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788744838.3064055_rf_level_1_lr_0.2_recording_lr_sweep_no_ffcv_test_acc_93.91.pth"

# resnet50 rf_level=4 lr=0.0001
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.0001_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788744840.882203_rf_level_4_lr_0.0001_recording_lr_sweep_no_ffcv_test_acc_69.64.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.0001_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788744898.7421641_rf_level_4_lr_0.0001_recording_lr_sweep_no_ffcv_test_acc_70.43.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.0001_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788745037.2133198_rf_level_4_lr_0.0001_recording_lr_sweep_no_ffcv_test_acc_68.4.pth"

# resnet50 rf_level=4 lr=0.001
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.001_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788745401.9234877_rf_level_4_lr_0.001_recording_lr_sweep_no_ffcv_test_acc_82.12.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.001_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788746722.6990306_rf_level_4_lr_0.001_recording_lr_sweep_no_ffcv_test_acc_82.45.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.001_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788747010.1366668_rf_level_4_lr_0.001_recording_lr_sweep_no_ffcv_test_acc_81.99.pth"

# resnet50 rf_level=4 lr=0.003
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.003_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788747796.701767_rf_level_4_lr_0.003_recording_lr_sweep_no_ffcv_test_acc_84.4.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.003_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788748038.89654_rf_level_4_lr_0.003_recording_lr_sweep_no_ffcv_test_acc_83.4.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.003_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788748593.7011423_rf_level_4_lr_0.003_recording_lr_sweep_no_ffcv_test_acc_83.74.pth"

# resnet50 rf_level=4 lr=0.005
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.005_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788749033.7146225_rf_level_4_lr_0.005_recording_lr_sweep_no_ffcv_test_acc_84.1.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.005_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788749539.8593116_rf_level_4_lr_0.005_recording_lr_sweep_no_ffcv_test_acc_81.18.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.005_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788749848.3423297_rf_level_4_lr_0.005_recording_lr_sweep_no_ffcv_test_acc_77.75.pth"

# resnet50 rf_level=4 lr=0.007
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.007_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788751625.7091773_rf_level_4_lr_0.007_recording_lr_sweep_no_ffcv_test_acc_83.13.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.007_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788751844.2021394_rf_level_4_lr_0.007_recording_lr_sweep_no_ffcv_test_acc_83.86.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.007_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788751860.8808124_rf_level_4_lr_0.007_recording_lr_sweep_no_ffcv_test_acc_84.23.pth"

# resnet50 rf_level=4 lr=0.1
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.1_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788752624.71632_rf_level_4_lr_0.1_recording_lr_sweep_no_ffcv_test_acc_89.7.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.1_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788752830.3645015_rf_level_4_lr_0.1_recording_lr_sweep_no_ffcv_test_acc_89.5.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.1_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788753283.4540498_rf_level_4_lr_0.1_recording_lr_sweep_no_ffcv_test_acc_89.73.pth"

# resnet50 rf_level=4 lr=0.2
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.2_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788754014.7215848_rf_level_4_lr_0.2_recording_lr_sweep_no_ffcv_test_acc_89.51.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.2_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788754115.6596978_rf_level_4_lr_0.2_recording_lr_sweep_no_ffcv_test_acc_89.08.pth"
run_pruning_finetune "resnet50" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.2_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "resnet50_normal_cifar10_1788754589.922838_rf_level_4_lr_0.2_recording_lr_sweep_no_ffcv_test_acc_89.32.pth"

# vgg19 rf_level=1 lr=0.0001
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.0001_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788712914.2861767_rf_level_1_lr_0.0001_recording_lr_sweep_no_ffcv_test_acc_82.56.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.0001_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788713022.946601_rf_level_1_lr_0.0001_recording_lr_sweep_no_ffcv_test_acc_80.82.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.0001_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788713117.0697205_rf_level_1_lr_0.0001_recording_lr_sweep_no_ffcv_test_acc_81.53.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.0001_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788715485.5843997_rf_level_1_lr_0.0001_recording_lr_sweep_no_ffcv_test_acc_82.87.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.0001_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788715503.2196443_rf_level_1_lr_0.0001_recording_lr_sweep_no_ffcv_test_acc_82.4.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.0001_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788715532.770455_rf_level_1_lr_0.0001_recording_lr_sweep_no_ffcv_test_acc_82.28.pth"

# vgg19 rf_level=1 lr=0.001
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.001_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788713209.7757642_rf_level_1_lr_0.001_recording_lr_sweep_no_ffcv_test_acc_88.04.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.001_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788715610.0865316_rf_level_1_lr_0.001_recording_lr_sweep_no_ffcv_test_acc_89.58.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.001_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788715730.759108_rf_level_1_lr_0.001_recording_lr_sweep_no_ffcv_test_acc_90.08.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.001_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788715800.7998536_rf_level_1_lr_0.001_recording_lr_sweep_no_ffcv_test_acc_90.1.pth"

# vgg19 rf_level=1 lr=0.003
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.003_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788715995.7084787_rf_level_1_lr_0.003_recording_lr_sweep_no_ffcv_test_acc_91.87.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.003_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788716748.5713255_rf_level_1_lr_0.003_recording_lr_sweep_no_ffcv_test_acc_91.91.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.003_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788716776.1972272_rf_level_1_lr_0.003_recording_lr_sweep_no_ffcv_test_acc_91.42.pth"

# vgg19 rf_level=1 lr=0.005
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.005_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788716861.0371158_rf_level_1_lr_0.005_recording_lr_sweep_no_ffcv_test_acc_92.27.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.005_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788717482.3884494_rf_level_1_lr_0.005_recording_lr_sweep_no_ffcv_test_acc_92.14.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.005_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788717492.499358_rf_level_1_lr_0.005_recording_lr_sweep_no_ffcv_test_acc_92.22.pth"

# vgg19 rf_level=1 lr=0.007
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.007_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788717512.386441_rf_level_1_lr_0.007_recording_lr_sweep_no_ffcv_test_acc_92.1.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.007_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788717540.3817_rf_level_1_lr_0.007_recording_lr_sweep_no_ffcv_test_acc_92.63.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.007_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788717635.3857872_rf_level_1_lr_0.007_recording_lr_sweep_no_ffcv_test_acc_92.47.pth"

# vgg19 rf_level=1 lr=0.1
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.1_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788717712.5394752_rf_level_1_lr_0.1_recording_lr_sweep_no_ffcv_test_acc_93.07.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.1_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788717731.6579177_rf_level_1_lr_0.1_recording_lr_sweep_no_ffcv_test_acc_92.9.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.1_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788717962.1773107_rf_level_1_lr_0.1_recording_lr_sweep_no_ffcv_test_acc_92.95.pth"

# vgg19 rf_level=1 lr=0.2
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.2_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788718163.7206402_rf_level_1_lr_0.2_recording_lr_sweep_no_ffcv_test_acc_92.97.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.2_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788718205.4628863_rf_level_1_lr_0.2_recording_lr_sweep_no_ffcv_test_acc_93.13.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.2_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 1 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788718632.2567356_rf_level_1_lr_0.2_recording_lr_sweep_no_ffcv_test_acc_92.9.pth"

# vgg19 rf_level=4 lr=0.0001
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.0001_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788718772.9758093_rf_level_4_lr_0.0001_recording_lr_sweep_no_ffcv_test_acc_71.18.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.0001_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788718785.3818936_rf_level_4_lr_0.0001_recording_lr_sweep_no_ffcv_test_acc_71.13.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.0001_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788718787.6567783_rf_level_4_lr_0.0001_recording_lr_sweep_no_ffcv_test_acc_71.11.pth"

# vgg19 rf_level=4 lr=0.001
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.001_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788719410.719972_rf_level_4_lr_0.001_recording_lr_sweep_no_ffcv_test_acc_80.89.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.001_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788719415.1342764_rf_level_4_lr_0.001_recording_lr_sweep_no_ffcv_test_acc_80.35.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.001_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788719438.0126386_rf_level_4_lr_0.001_recording_lr_sweep_no_ffcv_test_acc_80.47.pth"

# vgg19 rf_level=4 lr=0.003
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.003_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788719479.7494066_rf_level_4_lr_0.003_recording_lr_sweep_no_ffcv_test_acc_83.66.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.003_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788719719.0861125_rf_level_4_lr_0.003_recording_lr_sweep_no_ffcv_test_acc_83.95.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.003_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788719733.595991_rf_level_4_lr_0.003_recording_lr_sweep_no_ffcv_test_acc_84.03.pth"

# vgg19 rf_level=4 lr=0.005
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.005_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788719963.1658373_rf_level_4_lr_0.005_recording_lr_sweep_no_ffcv_test_acc_83.8.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.005_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788720116.3774176_rf_level_4_lr_0.005_recording_lr_sweep_no_ffcv_test_acc_84.66.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.005_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788720368.181866_rf_level_4_lr_0.005_recording_lr_sweep_no_ffcv_test_acc_84.14.pth"

# vgg19 rf_level=4 lr=0.007
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.007_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788720478.6662486_rf_level_4_lr_0.007_recording_lr_sweep_no_ffcv_test_acc_84.78.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.007_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788720601.4875767_rf_level_4_lr_0.007_recording_lr_sweep_no_ffcv_test_acc_84.44.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.007_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788720704.6770127_rf_level_4_lr_0.007_recording_lr_sweep_no_ffcv_test_acc_84.86.pth"

# vgg19 rf_level=4 lr=0.1
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.1_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788720834.7581613_rf_level_4_lr_0.1_recording_lr_sweep_no_ffcv_test_acc_85.46.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.1_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788721215.3549552_rf_level_4_lr_0.1_recording_lr_sweep_no_ffcv_test_acc_85.04.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.1_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788721245.219036_rf_level_4_lr_0.1_recording_lr_sweep_no_ffcv_test_acc_84.95.pth"

# vgg19 rf_level=4 lr=0.2
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.2_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788721326.9660506_rf_level_4_lr_0.2_recording_lr_sweep_no_ffcv_test_acc_79.37.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.2_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788721355.7628684_rf_level_4_lr_0.2_recording_lr_sweep_no_ffcv_test_acc_84.88.pth"
run_pruning_finetune "vgg19" "cifar10" "${checkpoints_folder}" "${SCRATCH}/data2" "${save_folder}" "lr_0.2_recording_lr_sweep_no_ffcv" 0 "" "" "${pruning_rate}" 4 "${resolution}" "${resize}" "vgg19_normal_cifar10_1788721706.5880888_rf_level_4_lr_0.2_recording_lr_sweep_no_ffcv_test_acc_79.88.pth"
