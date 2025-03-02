#!/bin/bash
#for lvl in 2; do                # iterate idxa from 0 to length
##for lvl in 8; do                # iterate idxa from 0 to length
#for model in "densenet40";do
#for dataset in "tiny_imagenet"; do
#sbatch --nodes=1 --time=140:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="${model}_lvl_${lvl}_${dataset}.err" --gres=gpu:1 --output="${model}_lvl_${lvl}_${dataset}.out"  --job-name="${model}_lvl_${lvl}_${dataset}" slurm_new_models_original_paper_run.sh "${model}" "${dataset}" 0 ${lvl}  "normal" 200 "recording_200" 1 1
#done
#done
#done

for lvl in 0; do                # iterate idxa from 0 to length
for model in  "resnet50";do
for dataset in "cifar10"; do

sbatch --nodes=1 --time=140:00:00 -t 1-5 --partition=gpu  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="${model}_lvl_${lvl}_${dataset}.err" --gres=gpu:1 --output="${model}_lvl_${lvl}_${dataset}.out"  --job-name="${model}_lvl_${lvl}_${dataset}" slurm_new_models_original_paper_run.sh "${model}" "${dataset}" 2 ${lvl}  "normal" 100 "recording_100" 1 1

done
done
done

for lvl in 0; do    # iterate idxa from 0 to length
for model in  "vgg19";do
for dataset in "cifar10" "tiny_imagenet"; do
sbatch --nodes=1 --time=140:00:00 -t 1-5--partition=gpu  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="${model}_lvl_${lvl}_${dataset}.err" --gres=gpu:1 --output="${model}_lvl_${lvl}_${dataset}.out"  --job-name="${model}_lvl_${lvl}_${dataset}" slurm_new_models_original_paper_run.sh "${model}" "${dataset}" 2 ${lvl}  "normal" 100 "recording_100" 1 1
done
done
done
