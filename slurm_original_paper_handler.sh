#!/bin/bash

# VGG
#sbatch --nodes=1 --time=10:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error=".err" --gres=gpu:1 --output="small_resnet_lv_3_small_imagenet_no_ffcv.output"  --job-name="small_resnet_lv_3_small_imagenet_no_ffcv" slurm_run.sh "resnet_small" "small_imagenet" 4 3 "normal" 200 "no_recording_200_no_ffcv" 1 1 "{}" #"/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv"
#sbatch --nodes=1 --time=10:00:00 --partition=small  --mail-type=all --mail-user=sclaam@leeds.ac.uk --error=".err" --gres=gpu:1 --output="small_resnet_lv_4_small_imagenet_no_ffcv.output"  --job-name="small_resnet_lv_4_small_imagenet_no_ffcv" slurm_run.sh "resnet_small" "small_imagenet" 4 4 "normal" 200 "no_recording_200_no_ffcv" 1 1 #"/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv"

single_run_paper_training() {
model=$1
dataset=$2
#RF_level=$3
num_workers=$3
name=$4
record=$5
save_folder=$6
epochs=$7
ffcv=$8
ffcv_train=$9
ffcv_val="${10}"
RF_level="${11}"

echo "model ${model} and dataset ${dataset}"

#pruning_rates=("0.5" "0.6" "0.7" "0.8" "0.9" "0.95")
if [ "${ffcv}" -eq 1 ]; then
    string_ffcv="ffcv"
else

    string_ffcv="no_ffcv"
fi

# For resnet18


if [ "${model}" == "vgg19" ]; then

    rf_levels=("1" "2" "3" "4")

fi

if [ "${model}" == "resnet50" ] && [ "${dataset}" == "cifar10" ]; then

    rf_levels=("1" "2" "3" "4" "9" "10" "11")
fi
if [ "${model}" == "resnet50" ] && [ "${dataset}" == "tiny_imagenet" ]; then

    rf_levels=("2" "3" "4" "9" "10" "11")

fi
#pruners=("lamp")

#pruners_max=${#pruners[@]}                                  # Take the length of that array
levels_max=${#rf_levels[@]}                                  # Take the length of that array
#number_pruning_rates=${#pruning_rates[@]}                            # Take the length of that array
#for ((idxA=0; idxA<number_pruning_rates; idxA++)); do                # iterate idxA from 0 to length
#for ((idxA=0; idxA<levels_max; idxA++));do              # iterate idxB from 0 to length
#for ((idxC=0; idxC<pruners_max; idxC++));do              # iterate idxB from 0 to length
#  echo "Entered the pruners loop"

#if [ $ffcv -eq 1 ]; then

sbatch --nodes=1 --time=10:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="train_${model}_${dataset}_${rf_levels[$idxA]}_${string_ffcv}.err" --output="train_${model}_${dataset}_${rf_levels[$idxA]}_${string_ffcv}.out" --job-name="train_${model}_${dataset}_${rf_levels[$idxA]}_${string_ffcv}" slurm_original_paper_run.sh FFCV="${ffcv}" NAME="${name}" MODEL="${model}" DATASET="${dataset}"  NUMW="${num_workers}"  RFL="${RF_level}" TYPE="normal" FOLDER="${save_folder}" EXPERIMENT=1 FFCV_TRAIN="${ffcv_train}" FFCV_VAL="${ffcv_val}" EPOCHS="${epochs}" RECORD="${record}" #DATA_FOLDER="${data_folder}" OUTPUT_DIR="${output_dir}" TOPK=10
#echo "train_${model}_${dataset}_${rf_levels[$idxA]}_${string_ffcv}"
#  sbatch --nodes=1 --time=10:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="train_${model}_${dataset}__${rf_levels[$idxB]}_no_ffcv.err" --output="train_${model}_${dataset}__${rf_levels[$idxB]}_no_ffcv.out" --job-name="train_${model}_${dataset}__${rf_levels[$idxB]}_no_ffcv" slurm_original_paper_run.sh


#done
#done
#done

}

run_paper_training() {
model=$1
dataset=$2
#RF_level=$3
num_workers=$3
name=$4
record=$5
save_folder=$6
epochs=$7
ffcv=$8
ffcv_train=$9
ffcv_val="${10}"

echo "model ${model} and dataset ${dataset}"

#pruning_rates=("0.5" "0.6" "0.7" "0.8" "0.9" "0.95")
if [ "${ffcv}" -eq 1 ]; then
    string_ffcv="ffcv"
else

    string_ffcv="no_ffcv"
fi

# For resnet18


if [ "${model}" == "vgg19" ]; then

    rf_levels=("1" "2" "3" "4")

fi

if [ "${model}" == "resnet50" ] && [ "${dataset}" == "cifar10" ]; then

    rf_levels=("1" "2" "3" "4" "9" "10" "11")
fi
if [ "${model}" == "resnet50" ] && [ "${dataset}" == "tiny_imagenet" ]; then

    rf_levels=("2" "3" "4" "9" "10" "11")

fi
#pruners=("lamp")

#pruners_max=${#pruners[@]}                                  # Take the length of that array
levels_max=${#rf_levels[@]}                                  # Take the length of that array
#number_pruning_rates=${#pruning_rates[@]}                            # Take the length of that array
#for ((idxA=0; idxA<number_pruning_rates; idxA++)); do                # iterate idxA from 0 to length
for ((idxA=0; idxA<levels_max; idxA++));do              # iterate idxB from 0 to length
#for ((idxC=0; idxC<pruners_max; idxC++));do              # iterate idxB from 0 to length
#  echo "Entered the pruners loop"

#if [ $ffcv -eq 1 ]; then

sbatch --nodes=1 --time=10:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="train_${model}_${dataset}_${rf_levels[$idxA]}_${string_ffcv}.err" --output="train_${model}_${dataset}_${rf_levels[$idxA]}_${string_ffcv}.out" --job-name="train_${model}_${dataset}_${rf_levels[$idxA]}_${string_ffcv}" slurm_original_paper_run.sh FFCV="${ffcv}" NAME="${name}" MODEL="${model}" DATASET="${dataset}"  NUMW="${num_workers}"  RFL="${rf_levels[$idxA]}" TYPE="normal" FOLDER="${save_folder}" EXPERIMENT=1 FFCV_TRAIN="${ffcv_train}" FFCV_VAL="${ffcv_val}" EPOCHS="${epochs}" RECORD="${record}" #DATA_FOLDER="${data_folder}" OUTPUT_DIR="${output_dir}" TOPK=10
#echo "train_${model}_${dataset}_${rf_levels[$idxA]}_${string_ffcv}"
#  sbatch --nodes=1 --time=10:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="train_${model}_${dataset}__${rf_levels[$idxB]}_no_ffcv.err" --output="train_${model}_${dataset}__${rf_levels[$idxB]}_no_ffcv.out" --job-name="train_${model}_${dataset}__${rf_levels[$idxB]}_no_ffcv" slurm_original_paper_run.sh


done
#done
#done

}

save_folder="${HOME}/original_paper_checkpoints"

single_run_paper_training "vgg19" "cifar10" "4" "recording_200_no_ffcv" "1" "${save_folder}" "200" "0" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv" "1"
#run_paper_training "vgg19" "cifar10" "4" "recording_200_no_ffcv" "1" "${save_folder}" "200" "0" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv"
run_paper_training "vgg19" "tiny_imagenet" "4" "recording_200_no_ffcv" "1" "${save_folder}" "200" "0" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv"
run_paper_training "resnet50" "cifar10" "4" "recording_200_no_ffcv" "1" "${save_folder}" "200" "0" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv"
run_paper_training "resnet50" "tiny_imagenet" "4" "recording_200_no_ffcv" "1" "${save_folder}" "200" "0" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv" "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv"
