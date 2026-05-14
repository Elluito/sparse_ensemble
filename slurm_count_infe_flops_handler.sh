#!/bin/bash

# Base folder where trained checkpoints live
CHECKPOINT_BASE="${SCRATCH}/FPGM_experiments_200_pr_0.8"

# Folder where inference-flops CSVs will be written
RESULTS_FOLDER="${SCRATCH}/inference_flops_results"

# ---------------------------------------------------------------
# resnet50 — cifar10
# ---------------------------------------------------------------
# 0.8 pruning rate

resolution=32
model="resnet50"
dataset="cifar10"
rf_lvls=(1 2 3 4 9 11)
seeds=(
    "resnet50_normal_cifar10_1778425133.3852606_rf_level_1_recording_fpgm_200_no_ffcv_pr_0.8_test_acc_84.86.pth"
    "resnet50_normal_cifar10_1778425133.4304452_rf_level_2_recording_fpgm_200_no_ffcv_pr_0.8_test_acc_83.07.pth"
    "resnet50_normal_cifar10_1778432995.5555873_rf_level_3_recording_fpgm_200_no_ffcv_pr_0.8_test_acc_80.99.pth"
    "resnet50_normal_cifar10_1778440737.4754174_rf_level_4_recording_fpgm_200_no_ffcv_pr_0.8_test_acc_77.12.pth"
    "resnet50_normal_cifar10_1778453230.8064427_rf_level_9_recording_fpgm_200_no_ffcv_pr_0.8_test_acc_60.77.pth"
    "resnet50_normal_cifar10_1778595950.771914_rf_level_11_recording_fpgm_200_no_ffcv_pr_0.8_test_acc_43.32.pth"
)
pruning_types=("fpmg" "normal")

  for i in "${!rf_lvls[@]}"; do
      rf_level="${rf_lvls[$i]}"
      checkpoint="${CHECKPOINT_BASE}/${seeds[$i]}"
      for j in "${!prunng_types[@]}";do
        p_type="${pruning_types[$j]}"

          if [ "${p_type}" == "fpmg" ]; then
            if [ -z "${checkpoint}" ]; then
              echo "WARNING: no checkpoint found for ${checkpoint}, skipping."
              continue
            fi

            sbatch --nodes=1 --time=00:15:00 --partition=small \
              --mail-type=all --mail-user=sclaam@leeds.ac.uk \
              --error="infe_flops_${model}_lvl_${rf_level}_${dataset}.err" \
              --output="infe_flops_${model}_lvl_${rf_level}_${dataset}.out" \
              --job-name="infe_flops_${model}_lvl_${rf_level}_${dataset}" \
              slurm_count_infe_flops_run.sh \
                "${model}" "${dataset}" 4 "${rf_level}" "normal" "infe_flops_${p_type}" 1 "${resolution}" \
                "${checkpoint}" "${RESULTS_FOLDER}" "${p_type}"

          fi

          if [ "${p_type}" == "normal" ]; then
            sbatch --nodes=1 --time=00:15:00 --partition=small \
              --mail-type=all --mail-user=sclaam@leeds.ac.uk \
              --error="infe_flops_${model}_lvl_${rf_level}_${dataset}.err" \
              --output="infe_flops_${model}_lvl_${rf_level}_${dataset}.out" \
              --job-name="infe_flops_${model}_lvl_${rf_level}_${dataset}" \
              slurm_count_infe_flops_run.sh \
                "${model}" "${dataset}" 4 "${rf_level}" "normal" "infe_flops_${p_type}" 1 "${resolution}" \
                "" "${RESULTS_FOLDER}" "${p_type}"

          fi


      done
  done



#for lvl in 1 2 3 4 9 11; do
#for model in "resnet50"; do
#for dataset in "cifar10"; do
#
#  checkpoint=$(find "${CHECKPOINT_BASE}" -name "${model}_normal_${dataset}_*rf_level_${lvl}_*.pth" | head -1)
#  if [ -z "${checkpoint}" ]; then
#    echo "WARNING: no checkpoint found for ${model} ${dataset} rf_level=${lvl}, skipping."
#    continue
#  fi
#
#  sbatch --nodes=1 --time=00:15:00 --partition=small \
#    --mail-type=all --mail-user=sclaam@leeds.ac.uk \
#    --error="infe_flops_${model}_lvl_${lvl}_${dataset}.err" \
#    --output="infe_flops_${model}_lvl_${lvl}_${dataset}.out" \
#    --job-name="infe_flops_${model}_lvl_${lvl}_${dataset}" \
#    slurm_count_infe_flops_run.sh \
#      "${model}" "${dataset}" 4 "${lvl}" "normal" "infe_flops" 1 "${resolution}" \
#      "${checkpoint}" "${RESULTS_FOLDER}"
#
#done
#done
#done
#
# ---------------------------------------------------------------
# resnet50 — tiny_imagenet
# ---------------------------------------------------------------
#resolution=64
#for lvl in 1 2 3 4 9 11; do
#for model in "resnet50"; do
#for dataset in "tiny_imagenet"; do
#
#  checkpoint=$(find "${CHECKPOINT_BASE}" -name "${model}_normal_${dataset}_*rf_level_${lvl}_*.pth" | head -1)
#  if [ -z "${checkpoint}" ]; then
#    echo "WARNING: no checkpoint found for ${model} ${dataset} rf_level=${lvl}, skipping."
#    continue
#  fi
#
#  sbatch --nodes=1 --time=00:15:00 --partition=small \
#    --mail-type=all --mail-user=sclaam@leeds.ac.uk \
#    --error="infe_flops_${model}_lvl_${lvl}_${dataset}.err" \
#    --output="infe_flops_${model}_lvl_${lvl}_${dataset}.out" \
#    --job-name="infe_flops_${model}_lvl_${lvl}_${dataset}" \
#    slurm_count_infe_flops_run.sh \
#      "${model}" "${dataset}" 4 "${lvl}" "normal" "infe_flops" 1 "${resolution}" \
#      "${checkpoint}" "${RESULTS_FOLDER}"
#
#done
#done
#done
#
# ---------------------------------------------------------------
# vgg19 — cifar10
# ---------------------------------------------------------------
#resolution=32
#for lvl in 1 2 3 4 9 11; do
#for model in "vgg19"; do
#for dataset in "cifar10"; do
#
#  checkpoint=$(find "${CHECKPOINT_BASE}" -name "${model}_normal_${dataset}_*rf_level_${lvl}_*.pth" | head -1)
#  if [ -z "${checkpoint}" ]; then
#    echo "WARNING: no checkpoint found for ${model} ${dataset} rf_level=${lvl}, skipping."
#    continue
#  fi
#
#  sbatch --nodes=1 --time=00:15:00 --partition=small \
#    --mail-type=all --mail-user=sclaam@leeds.ac.uk \
#    --error="infe_flops_${model}_lvl_${lvl}_${dataset}.err" \
#    --output="infe_flops_${model}_lvl_${lvl}_${dataset}.out" \
#    --job-name="infe_flops_${model}_lvl_${lvl}_${dataset}" \
#    slurm_count_infe_flops_run.sh \
#      "${model}" "${dataset}" 4 "${lvl}" "normal" "infe_flops" 1 "${resolution}" \
#      "${checkpoint}" "${RESULTS_FOLDER}"
#
#done
#done
#done

