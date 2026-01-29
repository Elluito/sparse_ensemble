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
for lvl in 91 180 269; do    # iterate idxa from 0 to length
for model in  "vgg19_dilation";do
for dataset in "cifar10"; do


#sbatch --nodes=1 --time=47:00:00 --array=1-5 --partition=gpu  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="${model}_lvl_${lvl}_${dataset}.err" --gres=gpu:1 --output="${model}_lvl_${lvl}_${dataset}.out"  --job-name="${model}}_dilation_lvl_${lvl}_${dataset}" slurm_dilation_experiments_run.sh "${model}" "${dataset}" 8 ${lvl}  "normal" 200 "recording_dilation_200_no_ffcv" 1 1 ${resolution}

sbatch --nodes=1 --time=47:00:00 --array=1-5 --partition=gpu  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="dilation_${model}_lvl_${lvl}_${dataset}.err" --gres=gpu:1 --output="dilation_${model}_lvl_${lvl}_${dataset}.out"  --job-name="dilation_${model}_lvl_${lvl}_${dataset}" slurm_dilation_experiments_run.sh "${model}" "${dataset}" 8 ${lvl}  "normal" 100 "recording_dilation_100_no_ffcv" 1 1 ${resolution}
done
done
done

resolution=64
for lvl in 91 180 269; do    # iterate idxa from 0 to length
for model in  "vgg19_dilation";do
for dataset in "tiny_imagenet"; do

#sbatch --nodes=1 --time=47:00:00 --array=1-5 --partition=gpu  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="${model}_lvl_${lvl}_${dataset}.err" --gres=gpu:1 --output="${model}_lvl_${lvl}_${dataset}.out"  --job-name="${model}_dilation_lvl_${lvl}_${dataset}" slurm_dilation_experiments_run.sh "${model}" "${dataset}" 8 ${lvl}  "normal" 200 "recording_dilation_200_no_ffcv" 1 1 ${resolution}

sbatch --nodes=1 --time=47:00:00 --array=1-5 --partition=gpu  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="dilation_${model}_lvl_${lvl}_${dataset}.err" --gres=gpu:1 --output="dilation_${model}_lvl_${lvl}_${dataset}.out"  --job-name="dilation_${model}_lvl_${lvl}_${dataset}" slurm_dilation_experiments_run.sh "${model}" "${dataset}" 8 ${lvl}  "normal" 100 "recording_dilation_100_no_ffcv" 1 1 ${resolution}

done
done
done


# RESNET50

resolution=32
for lvl in 54 107 159; do    # iterate idxa from 0 to length
for model in  "resnet50_dilation";do
for dataset in "cifar10"; do


#sbatch --nodes=1 --time=47:00:00 --array=1-5 --partition=gpu  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="${model}_lvl_${lvl}_${dataset}.err" --gres=gpu:1 --output="${model}_lvl_${lvl}_${dataset}.out"  --job-name="${model}}_dilation_lvl_${lvl}_${dataset}" slurm_dilation_experiments_run.sh "${model}" "${dataset}" 8 ${lvl}  "normal" 200 "recording_dilation_200_no_ffcv" 1 1 ${resolution}

sbatch --nodes=1 --time=47:00:00 --array=1-5 --partition=gpu  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="dilation_${model}_lvl_${lvl}_${dataset}.err" --gres=gpu:1 --output="dilation_${model}_lvl_${lvl}_${dataset}.out"  --job-name="dilation_${model}_lvl_${lvl}_${dataset}" slurm_dilation_experiments_run.sh "${model}" "${dataset}" 8 ${lvl}  "normal" 100 "recording_dilation_100_no_ffcv" 1 1 ${resolution}
done
done
done

resolution=64
for lvl in 54 107 159; do    # iterate idxa from 0 to length
for model in  "resnet50_dilation";do
for dataset in "tiny_imagenet"; do

#sbatch --nodes=1 --time=47:00:00 --array=1-5 --partition=gpu  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="${model}_lvl_${lvl}_${dataset}.err" --gres=gpu:1 --output="${model}_lvl_${lvl}_${dataset}.out"  --job-name="${model}_dilation_lvl_${lvl}_${dataset}" slurm_dilation_experiments_run.sh "${model}" "${dataset}" 8 ${lvl}  "normal" 200 "recording_dilation_200_no_ffcv" 1 1 ${resolution}

sbatch --nodes=1 --time=47:00:00 --array=1-5 --partition=gpu  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error="dilation_${model}_lvl_${lvl}_${dataset}.err" --gres=gpu:1 --output="dilation_${model}_lvl_${lvl}_${dataset}.out"  --job-name="dilation_${model}_lvl_${lvl}_${dataset}" slurm_dilation_experiments_run.sh "${model}" "${dataset}" 8 ${lvl}  "normal" 100 "recording_dilation_100_no_ffcv" 1 1 ${resolution}

done
done
done
