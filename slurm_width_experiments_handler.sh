#!/bin/bash




resolution=32
for lvl in 2 3 9 11; do    # iterate idxa from 0 to length
for width in 1 2 3; do
for model in  "resnet50";do
for dataset in "cifar10"; do

sbatch --nodes=1 --time=47:00:00 --array=1-5 --partition=gpu  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="width_${width}_${model}_lvl_${lvl}_${dataset}.err" --gres=gpu:1 --output="width_${width}_${model}_lvl_${lvl}_${dataset}.out"  --job-name="width_${width}_${model}_lvl_${lvl}_${dataset}" slurm_width_experiments_run.sh "${model}" "${dataset}" 8 ${lvl}  "normal" 200 "recording_width_${width}_200_no_ffcv" ${width} 1 ${resolution}


done
done
done
done
