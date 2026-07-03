#!/bin/bash

# VGG19 dilation + maxpool

resolution=32
for lvl in 1; do    # iterate idxa from 0 to length
for model in  "vgg19_dilation_max_pool";do
for dataset in "cifar10"; do

sbatch --nodes=1 --time=47:00:00 --array=1-5 --partition=gpu  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="dilation_max_pool_${model}_lvl_${lvl}_${dataset}.err" --gres=gpu:1 --output="dilation_max_pool_${model}_lvl_${lvl}_${dataset}.out"  --job-name="dilation_max_pool_${model}_lvl_${lvl}_${dataset}" slurm_dilation_max_pool_experiments_run.sh "${model}" "${dataset}" 8 ${lvl}  "normal" 100 "recording_dilation_max_pool_100_no_ffcv" 1 1 ${resolution}
done
done
done

resolution=64
for lvl in 1; do    # iterate idxa from 0 to length
for model in  "vgg19_dilation_max_pool";do
for dataset in "tiny_imagenet"; do

sbatch --nodes=1 --time=47:00:00 --array=1-5 --partition=gpu  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="dilation_max_pool_${model}_lvl_${lvl}_${dataset}.err" --gres=gpu:1 --output="dilation_max_pool_${model}_lvl_${lvl}_${dataset}.out"  --job-name="dilation_max_pool_${model}_lvl_${lvl}_${dataset}" slurm_dilation_max_pool_experiments_run.sh "${model}" "${dataset}" 8 ${lvl}  "normal" 100 "recording_dilation_max_pool_100_no_ffcv" 1 1 ${resolution}

done
done
done


# RESNET50 dilation + maxpool

resolution=32
for lvl in 1; do    # iterate idxa from 0 to length
for model in  "resnet50_dilation_max_pool";do
for dataset in "cifar10"; do

sbatch --nodes=1 --time=47:00:00 --array=1-5 --partition=gpu  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="dilation_max_pool_${model}_lvl_${lvl}_${dataset}.err" --gres=gpu:1 --output="dilation_max_pool_${model}_lvl_${lvl}_${dataset}.out"  --job-name="dilation_max_pool_${model}_lvl_${lvl}_${dataset}" slurm_dilation_max_pool_experiments_run.sh "${model}" "${dataset}" 8 ${lvl}  "normal" 100 "recording_dilation_max_pool_100_no_ffcv" 1 1 ${resolution}
done
done
done

resolution=64
for lvl in 1; do    # iterate idxa from 0 to length
for model in  "resnet50_dilation_max_pool";do
for dataset in "tiny_imagenet"; do

sbatch --nodes=1 --time=47:00:00 --array=1-5 --partition=gpu  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="dilation_max_pool_${model}_lvl_${lvl}_${dataset}.err" --gres=gpu:1 --output="dilation_max_pool_${model}_lvl_${lvl}_${dataset}.out"  --job-name="dilation_max_pool_${model}_lvl_${lvl}_${dataset}" slurm_dilation_max_pool_experiments_run.sh "${model}" "${dataset}" 8 ${lvl}  "normal" 100 "recording_dilation_max_pool_100_no_ffcv" 1 1 ${resolution}

done
done
done
