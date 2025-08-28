#!/bin/bash
#for lvl in 2; do                # iterate idxa from 0 to length
##for lvl in 8; do                # iterate idxa from 0 to length
#for model in "densenet40";do
#for dataset in "tiny_imagenet"; do
#sbatch --nodes=1 --time=140:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="${model}_lvl_${lvl}_${dataset}.err" --gres=gpu:1 --output="${model}_lvl_${lvl}_${dataset}.out"  --job-name="${model}_lvl_${lvl}_${dataset}" slurm_new_models_original_paper_run.sh "${model}" "${dataset}" 0 ${lvl}  "normal" 200 "recording_200" 1 1
#done
#done
#done
#resolution=32
#for lvl in 0; do                # iterate idxa from 0 to length
#for model in  "resnet50";do
#for dataset in "cifar10"; do
#
#sbatch --nodes=1 --time=47:00:00 --array=1-5 --partition=gpu  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="${model}_lvl_${lvl}_${dataset}.err" --gres=gpu:1 --output="${model}_lvl_${lvl}_${dataset}.out"  --job-name="${model}_lvl_${lvl}_${dataset}" slurm_new_models_original_paper_run.sh "${model}" "${dataset}" 8 ${lvl}  "normal" 200 "recording_200_no_ffcv" 1 1 ${resolution}
#
#done
#done
#done

resolution=32
save_folder="${SCRATCH}/second_order_checkpoints_changed_lr"

for lvl in 1 2 3 4; do    # iterate idxa from 0 to length
for model in  "vgg19";do
for dataset in "cifar10"; do
#sbatch --nodes=1 --time=47:00:00 --array=1-5 --partition=gpu  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="${model}_lvl_${lvl}_${dataset}.err" --gres=gpu:1 --output="${model}_lvl_${lvl}_${dataset}.out"  --job-name="${model}_lvl_${lvl}_${dataset}" slurm_new_models_original_paper_run.sh "${model}" "${dataset}" 8 ${lvl}  "normal" 200 "recording_200_no_ffcv" 1 1 ${resolution}
sbatch --nodes=1 --time=47:00:00 --partition=gpu  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="sgd_${model}_lvl_${lvl}_${dataset}.err" --gres=gpu:1 --output="sgd_${model}_lvl_${lvl}_${dataset}.out"  --job-name="sgd_${model}_lvl_${lvl}_${dataset}" slurm_sgd_train_run.sh "${model}" "${dataset}" 8 "0.01" ${lvl}  "normal" 200 "recording_200_no_ffcv_sgd_lr_0.01_m_0.9" 1 1 ${resolution} ${save_folder}
done
done
done
