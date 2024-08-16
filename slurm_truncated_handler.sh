#!/bin/bash
# Submission script for serial Python job
# Luis Alfredo Avendano  2021-10-22

# Run from the current directory and with current environment
#$ -V -cwd

# Ask for some time (hh:mm:ss max of 00:10:00)
#$ -l h_rt=07:00:00

# ASk for some GPU
# -l coproc_p100=1

# Ask for some memory (by default, 1G, without a request)
# -l node_type=40core-768G
# -l h_vmem=600G
#$ -l h_vmem=120G
# -t 2-5
# -pe smp 3
# Send emails when job starts and ends
#$ -m be
#export OMP_NUM_THREADS=3
#export MKL_NUM_THREADS=3
#export OPENBLAS_NUM_THREADS=1
# Now run the job
#module load intel openmpi
#module add anaconda
#module add cuda/11.1.1

#conda activate work
#which python
#unset GOMP_CPU_AFFINITY
#unset KMP_AFFINITY


CHECK_1="$HOME/checkpoints_temp"
CHECK_2="$HOME/checkpoints_arc4_2/checkpoints"
CHECK_3="$HOME/additional_checkpoints"


## Level 3  Models

#sbatch --nodes=1 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="logistic_probes_lvl3_s_1_128.err" --output="logistic_probes_lvl3_s_1_128.out"  --job-name="logistic_probes_lvl3_s_1_128" slurm_truncated_run.sh "resnet_small" "small_imagenet" "${CHECK_1}/resnet_small_normal_small_imagenet_seed.2_rf_level_3_recording_200_no_ffcv_test_acc_58.11.pth" "seed_2" 3 "normal" "${HOME}/truncated_models_results" 50 "0.001" 0
#sbatch --nodes=1 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="logistic_probes_lvl3_s_1_128.err" --output="logistic_probes_lvl3_s_1_128.out"  --job-name="logistic_probes_lvl3_s_1_128" slurm_truncated_run.sh "resnet_small" "small_imagenet" "${CHECK_3}/resnet_small_normal_small_imagenet_seed.0_rf_level_3_no_recording_200_no_ffcv_test_acc_61.44.pth" "seed_0" 3 "normal" "${HOME}/truncated_models_results" 50 "0.001" 0
#
## Level 4  Model
#sbatch --nodes=1 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="logistic_probes_lvl4_s_2_128.err" --output="logistic_probes_lvl4_s_2_128.out"  --job-name="logistic_probes_lvl4_s_2_128" slurm_truncated_run.sh "resnet_small" "small_imagenet" "${CHECK_1}/resnet_small_normal_small_imagenet_seed.2_rf_level_4_recording_200_ffcv_test_acc_60.31.pth" "seed_2" 4 "normal" "${HOME}/truncated_models_results" 50 "0.001" 0
#sbatch --nodes=1 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="logistic_probes_lvl4_s_2_128.err" --output="logistic_probes_lvl4_s_2_128.out"  --job-name="logistic_probes_lvl4_s_2_128" slurm_truncated_run.sh "resnet_small" "small_imagenet" "${CHECK_3}/resnet_small_normal_small_imagenet_seed.0_rf_level_4_no_recording_200_no_ffcv_test_acc_62.87.pth" "seed_0" 4 "normal" "${HOME}/truncated_models_results" 50 "0.001" 0
#
## Level 5  Models
#
#sbatch --nodes=1 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="logistic_probes_lvl5_s_1.err" --output="logistic_probes_lvl5_s_1.out"  --job-name="logistic_probes_lvl5_s_1" slurm_truncated_run.sh "resnet_small" "small_imagenet" "${CHECK_2}/resnet_small_normal_small_imagenet_seed.5_rf_level_5_recording_200_test_acc_61.97.pth" "seed_5" 5 "normal" "${HOME}/truncated_models_results" 50 "0.001" 0
#
# Level 6  Models

#sbatch --nodes=1 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="logistic_probes_lvl6_s_1_128.err" --output="logistic_probes_lvl6_s_1_128.out"  --job-name="logistic_probes_lvl6_s_1_128" slurm_truncated_run.sh "resnet_small" "small_imagenet" "${CHECK_1}/resnet_small_normal_small_imagenet_seed.3_rf_level_6_recording_200_ffcv_test_acc_57.83.pth" "seed_3" 6 "normal" "${HOME}/truncated_models_results" 50 "0.001" 0
#sbatch --nodes=1 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="logistic_probes_lvl6_s_1_128.err" --output="logistic_probes_lvl6_s_1_128.out"  --job-name="logistic_probes_lvl6_s_1_128" slurm_truncated_run.sh "resnet_small" "small_imagenet" "${CHECK_3}/resnet_small_normal_small_imagenet_seed.1_rf_level_6_no_recording_200_no_ffcv_test_acc_59.49.pth" "seed_1" 6 "normal" "${HOME}/truncated_models_results" 50 "0.001" 0

# Level 7  Models

#sbatch --nodes=1 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="logistic_probes_lvl7_s_1.err" --output="logistic_probes_lvl7_s_1.out"  --job-name="logistic_probes_lvl7_s_1" slurm_truncated_run.sh "resnet_small" "small_imagenet" "${CHECK_2}/resnet_small_normal_small_imagenet_seed.9_rf_level_7_recording_200_test_acc_51.51.pth" "seed_9" 7 "normal" "${HOME}/truncated_models_results" 50 "0.001" 0
#
## Level 8  Models
#
#sbatch --nodes=1 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="logistic_probes_lvl8_s_1.err" --output="logistic_probes_lvl8_s_1.out"  --job-name="logistic_probes_lvl8_s_1" slurm_truncated_run.sh "resnet_small" "small_imagenet" "${CHECK_2}/resnet_small_normal_small_imagenet_seed.4_rf_level_8_recording_200_test_acc_49.1.pth" "seed_4" 8 "normal" "${HOME}/truncated_models_results" 50 "0.001" 0
#
#
## Level 9  Models
#
#sbatch --nodes=1 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="logistic_probes_lvl9_s_1.err" --output="logistic_probes_lvl9_s_1.out"  --job-name="logistic_probes_lvl9_s_1" slurm_truncated_run.sh "resnet_small" "small_imagenet" "${CHECK_2}/resnet_small_normal_small_imagenet_seed.5_rf_level_9_recording_200_test_acc_45.39.pth" "seed_5" 9 "normal" "${HOME}/truncated_models_results" 50 "0.001" 0
#
## Level 10  Models
#
#sbatch --nodes=1 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="logistic_probes_lvl10_s_1_128.err" --output="logistic_probes_lvl10_s_1_64.out"  --job-name="logistic_probes_lvl10_s_1_64" slurm_truncated_run.sh "resnet_small" "small_imagenet" "${CHECK_1}/resnet_small_normal_small_imagenet_seed.1_rf_level_10_recording_200_ffcv_test_acc_40.4.pth" "seed_1" 10 "normal" "${HOME}/truncated_models_results" 50 "0.001" 0
#sbatch --nodes=1 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="logistic_probes_lvl10_s_1_128.err" --output="logistic_probes_lvl10_s_1_64.out"  --job-name="logistic_probes_lvl10_s_1_64" slurm_truncated_run.sh "resnet_small" "small_imagenet" "${CHECK_3}/./resnet_small_normal_small_imagenet_seed.1_rf_level_10_no_recording_200_no_ffcv_test_acc_41.01.pth" "seed_1" 10 "normal" "${HOME}/truncated_models_results" 50 "0.001" 0


run_paper_truncated() {
model=$1
dataset=$2
RF_level=$3
#num_workers=$4
name=$4
solution=$5
#record=$6$9
save_folder=$6
epochs=$7
lr=$8
ffcv=$9

echo "model ${model} and dataset ${dataset}"

#pruning_rates=("0.5" "0.6" "0.7" "0.8" "0.9" "0.95")
#if [ "${ffcv}" -eq 1 ]; then
#    string_ffcv="ffcv"
#else
#
#    string_ffcv="no_ffcv"
#fi
#
# For resnet18

#
#if [ "${model}" == "vgg19" ]; then
#
#    rf_levels=("1" "2" "3" "4")
#
#fi
#
#if [ "${model}" == "resnet50" ] && [ "${dataset}" == "cifar10" ]; then
#
#    rf_levels=("1" "2" "3" "4" ")
#fi
#if [ "${model}" == "resnet50" ] && [ "${dataset}" == "tiny_imagenet" ]; then
#
#    rf_levels=("2" "3" "4" "9" "10" "11")
#
#fi
#pruners=("lamp")

#pruners_max=${#pruners[@]}                                  # Take the length of that array
#levels_max=${#rf_levels[@]}                                  # Take the length of that array
#number_pruning_rates=${#pruning_rates[@]}                            # Take the length of that array
#for ((idxA=0; idxA<number_pruning_rates; idxA++)); do                # iterate idxA from 0 to length
#for ((idxA=0; idxA<levels_max; idxA++));do              # iterate idxB from 0 to length
#for ((idxC=0; idxC<pruners_max; idxC++));do              # iterate idxB from 0 to length
#  echo "Entered the pruners loop"

#if [ $ffcv -eq 1 ]; then

#sbatch --nodes=1 --time=10:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="train_${model}_${dataset}_${rf_levels[$idxA]}_${string_ffcv}.err" --output="train_${model}_${dataset}_${rf_levels[$idxA]}_${string_ffcv}.out" --job-name="train_${model}_${dataset}_${rf_levels[$idxA]}_${string_ffcv}" slurm_original_paper_run.sh FFCV="${ffcv}" NAME="${name}" MODEL="${model}" DATASET="${dataset}"  NUMW="${num_workers}"  RFL="${rf_levels[$idxA]}" TYPE="normal" FOLDER="${save_folder}" EXPERIMENT=1 FFCV_TRAIN="${ffcv_train}" FFCV_VAL="${ffcv_val}" EPOCHS="${epochs}" RECORD="${record}" #DATA_FOLDER="${data_folder}" OUTPUT_DIR="${output_dir}" TOPK=10
#echo "train_${model}_${dataset}_${rf_levels[$idxA]}_${string_ffcv}"
#  sbatch --nodes=1 --time=10:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="train_${model}_${dataset}__${rf_levels[$idxB]}_no_ffcv.err" --output="train_${model}_${dataset}__${rf_levels[$idxB]}_no_ffcv.out" --job-name="train_${model}_${dataset}__${rf_levels[$idxB]}_no_ffcv" slurm_original_paper_run.sh

#name="${model}_${dataset}_lvl${RF_level}"
sbatch --nodes=1 --time=140:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="logistic_probes_${name}.err" --output="logistic_probes_${name}.out"  --job-name="logistic_probes_${name}" slurm_truncated_run.sh "${model}" "${dataset}" "${solution}" "${name}" "${RF_level}" "normal" "${save_folder}" "${epochs}" "${lr}" "${ffcv}"

#done
#done
#done

}

# CIFAR 10
## vgg
save_folder="${HOME}/truncated_models_results"
sol1vgg=/jmain02/home/J2AD014/mtc03/lla98-mtc03/original_paper_checkpoints/vgg19_normal_cifar10_seed.0_rf_level_1_recording_200_no_ffcv_test_acc_93.77.pth
sol2vgg=/jmain02/home/J2AD014/mtc03/lla98-mtc03/original_paper_checkpoints/vgg19_normal_cifar10_seed.0_rf_level_2_recording_200_no_ffcv_test_acc_90.98.pth
sol3vgg=/jmain02/home/J2AD014/mtc03/lla98-mtc03/original_paper_checkpoints/vgg19_normal_cifar10_seed.0_rf_level_3_recording_200_no_ffcv_test_acc_88.13.pth
sol4vgg=/jmain02/home/J2AD014/mtc03/lla98-mtc03/original_paper_checkpoints/vgg19_normal_cifar10_seed.0_rf_level_4_recording_200_no_ffcv_test_acc_86.12.pth
# Resnet50 CIFAR10
sol1rs50=/jmain02/home/J2AD014/mtc03/lla98-mtc03/original_paper_checkpoints/resnet50_normal_cifar10_seed.0_rf_level_1_recording_200_no_ffcv_test_acc_94.99.pth
sol2rs50=/jmain02/home/J2AD014/mtc03/lla98-mtc03/original_paper_checkpoints/resnet50_normal_cifar10_seed.0_rf_level_2_recording_200_no_ffcv_test_acc_94.24.pth
sol3rs50=/jmain02/home/J2AD014/mtc03/lla98-mtc03/original_paper_checkpoints/resnet50_normal_cifar10_seed.0_rf_level_3_recording_200_no_ffcv_test_acc_92.3.pth
sol4rs50=/jmain02/home/J2AD014/mtc03/lla98-mtc03/original_paper_checkpoints/resnet50_normal_cifar10_seed.0_rf_level_4_recording_200_no_ffcv_test_acc_90.91.pth
sol5rs50=/jmain02/home/J2AD014/mtc03/lla98-mtc03/original_paper_checkpoints/resnet50_normal_cifar10_seed.0_rf_level_9_recording_200_no_ffcv_test_acc_73.45.pth
sol6rs50=/jmain02/home/J2AD014/mtc03/lla98-mtc03/original_paper_checkpoints/resnet50_normal_cifar10_seed.0_rf_level_10_recording_200_no_ffcv_test_acc_53.77.pth
sol7rs50=/jmain02/home/J2AD014/mtc03/lla98-mtc03/original_paper_checkpoints/resnet50_normal_cifar10_seed.0_rf_level_11_recording_200_no_ffcv_test_acc_50.06.pth






rf_levels=("1" "2" "3" "4")
solutions=("${sol1vgg}" "${sol2vgg}" "${sol3vgg}" "${sol4vgg}")
names=("seed_0" "seed_0" "seed_0" "seed_0")
levels_max=${#rf_levels[@]}                                  # Take the length of that array
for ((idxA=0; idxA<levels_max; idxA++));do              # iterate idxB from 0 to length
run_paper_truncated "vgg19" "cifar10" "${rf_levels[$idxA]}" "${names[$idxA]}" "${solutions[$idxA]}" "${save_folder}" "50" "0.001" "0"
done


## RESNET50


rf_levels=("1" "2" "3" "4" "9" "10" "11")
solutions=("${sol1rs50}" "${sol2rs50}" "${sol3rs50}" "${sol4rs50}" "${sol5rs50}" "${sol6rs50}" "${sol7rs50}")
#names=("seed_0" "seed_0" "seed_0" "seed_0" "seed_0" "seed_0" "seed_0")
names=("seed_0" "seed_0" "seed_0" "seed_0" "seed_0" "seed_0" "seed_0")
levels_max=${#rf_levels[@]}                                  # Take the length of that array
for ((idxA=0; idxA<levels_max; idxA++));do              # iterate idxB from 0 to length
run_paper_truncated "resnet50" "cifar10" "${rf_levels[$idxA]}" "${names[$idxA]}" "${solutions[$idxA]}" "${save_folder}" "50" "0.001" "0"
done
# Tiny IMagenet
## vgg

## RESNET50
