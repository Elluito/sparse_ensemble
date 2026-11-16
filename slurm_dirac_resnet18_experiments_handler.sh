#!/bin/bash
# CIFAR10
# Controls for the diverse pooling analysis: does the dependence of the saturation effect on kernel size come from the
# residual connections (DiracNet-50: same depth / feature-map sizes as ResNet-50, no skip connections) or from depth
# (ResNet-18)? 10 seeds (array jobs) per kernel size, 100 epochs, lr 0.1, saturation recorded every epoch.
# Each array task gets its own seed name from train_CIFAR10.py (time stamp), as in the diverse pooling runs.

resolution=32
epochs=100
lr=0.1
record_saturation=1

## DiracNet-50

save_folder="${SCRATCH}/dirac_models_experiments"
for lvl in 1 2 3 4; do    # iterate rf_level from 1 to 4
for model in "diracnet50"; do
for dataset in "cifar10"; do

sbatch --nodes=1 --time=47:00:00 --array=1-10 --partition=gpu  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="dirac_${model}_lvl_${lvl}_${dataset}_seed_%a.err" --gres=gpu:1 --output="dirac_${model}_lvl_${lvl}_${dataset}_seed_%a.out"  --job-name="dirac_${model}_lvl_${lvl}_${dataset}" slurm_dirac_resnet18_experiments_run.sh "${model}" "${dataset}" 8 ${lvl}  "normal" ${epochs} "recording_dirac_100_no_ffcv" 1 1 ${resolution} "${save_folder}" ${lr} ${record_saturation}
done
done
done


## ResNet-18

save_folder="${SCRATCH}/resnet18_models_experiments"
for lvl in 1 2 3 4; do    # iterate rf_level from 1 to 4
for model in "resnet18"; do
for dataset in "cifar10"; do

sbatch --nodes=1 --time=47:00:00 --array=1-10 --partition=gpu  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="resnet18_${model}_lvl_${lvl}_${dataset}_seed_%a.err" --gres=gpu:1 --output="resnet18_${model}_lvl_${lvl}_${dataset}_seed_%a.out"  --job-name="resnet18_${model}_lvl_${lvl}_${dataset}" slurm_dirac_resnet18_experiments_run.sh "${model}" "${dataset}" 8 ${lvl}  "normal" ${epochs} "recording_resnet18_100_no_ffcv" 1 1 ${resolution} "${save_folder}" ${lr} ${record_saturation}
done
done
done
