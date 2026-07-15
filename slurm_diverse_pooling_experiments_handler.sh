#!/bin/bash
# CIFAR10
## VGG19 diverse pooling (spectral_pool, softpool, mixedpool, lippool)

resolution=32
for lvl in 1 2 3 4; do    # iterate rf_level from 1 to 4
for model in "vgg19_spectral_pool" "vgg19_softpool" "vgg19_mixedpool" "vgg19_lippool";do
for dataset in "cifar10"; do

sbatch --nodes=1 --time=47:00:00 --array=1-3 --partition=gpu  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="diverse_pooling_${model}_lvl_${lvl}_${dataset}.err" --gres=gpu:1 --output="diverse_pooling_${model}_lvl_${lvl}_${dataset}.out"  --job-name="diverse_pooling_${model}_lvl_${lvl}_${dataset}" slurm_diverse_pooling_experiments_run.sh "${model}" "${dataset}" 8 ${lvl}  "normal" 100 "recording_diverse_pooling_100_no_ffcv" 1 1 ${resolution}
done
done
done


## RESNET50 diverse pooling (spectral_pool, softpool, mixedpool, lippool)

resolution=32
for lvl in 1 2 3 4; do    # iterate rf_level from 1 to 4
for model in "resnet50_spectral_pool" "resnet50_softpool" "resnet50_mixedpool" "resnet50_lippool";do
for dataset in "cifar10"; do

sbatch --nodes=1 --time=47:00:00 --array=1-3 --partition=gpu  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="diverse_pooling_${model}_lvl_${lvl}_${dataset}.err" --gres=gpu:1 --output="diverse_pooling_${model}_lvl_${lvl}_${dataset}.out"  --job-name="diverse_pooling_${model}_lvl_${lvl}_${dataset}" slurm_diverse_pooling_experiments_run.sh "${model}" "${dataset}" 8 ${lvl}  "normal" 100 "recording_diverse_pooling_100_no_ffcv" 1 1 ${resolution}
done
done
done
