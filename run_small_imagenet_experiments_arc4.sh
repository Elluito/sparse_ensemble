#!/bin/bash -l
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
conda activate work
which python
unset GOMP_CPU_AFFINITY
unset KMP_AFFINITY


#qsub -t 1-4 -l h_rt=44:00:00 -l coproc_v100=1  -N "training_Level_5_resnet_small_small_imagenet" run.sh "resnet_small" "small_imagenet" 4 5 "normal" 200 "recording_200" 1 1 "/nobackup/sclaam/checkpoints"
#qsub -t 1-4 -l h_rt=44:00:00 -l coproc_v100=1  -N "training_Level_7_resnet_small_small_imagenet" run.sh "resnet_small" "small_imagenet" 4 7 "normal" 200 "recording_200" 1 1 "/nobackup/sclaam/checkpoints"
#qsub -t 1-4 -l h_rt=44:00:00 -l coproc_v100=1  -N "training_Level_8_resnet_small_small_imagenet" run.sh "resnet_small" "small_imagenet" 4 8  "normal" 200 "recording_200" 1 1 "/nobackup/sclaam/checkpoints"
#qsub -t 1-4 -l h_rt=44:00:00 -l coproc_v100=1  -N "training_Level_9_resnet_small_small_imagenet" run.sh "resnet_small" "small_imagenet" 4 9  "normal" 200 "recording_200" 1 1 "/nobackup/sclaam/checkpoints"

#"resnet_small" "small_imagenet" 4 10 "normal" 200 "recording_200_no_ffcv" 1 1


######################## Pruning one shot results ##############################

run_pruning() {
model=$1
dataset=$2
directory=$3
name=$4
ffcv=$5
ffcv_train=$6
ffcv_val=$7
echo "model ${model} and dataset ${dataset}"

pruning_rates=("0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9")
# For resnet18

  if [ "${8}" -gt 0 ]
  then
      rf_levels=("5" "7" "8" "9")
#       rf_levels=("10")

  else
        rf_levels=( "4")
  fi

levels_max=${#rf_levels[@]}                                  # Take the length of that array
number_pruning_rates=${#pruning_rates[@]}                            # Take the length of that array
for ((idxA=0; idxA<number_pruning_rates; idxA++)); do                # iterate idxA from 0 to length
for ((idxB=0; idxB<levels_max; idxB++));do              # iterate idxB from 0 to length


  if [ "${ffcv}" -gt 0 ]
  then
  echo "Use FFCV"
#sbatch --nodes=1 --time=03:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk  --error="${model}_${rf_levels[$idxB]}_${dataset}_${pruning_rates[$idxA]}_pruning_ffcv.err" --output="${model}_${rf_levels[$idxB]}_${dataset}_${pruning_rates[$idxA]}_pruning_ffcv.out" --job-name="${model}_${rf_levels[$idxB]}_${dataset}_${pruning_rates[$idxA]}_pruning_ffcv"   slurm_pruning_run.sh FFCV="${ffcv}" NAME="${name}" MODEL="${model}" DATASET="${dataset}"  NUMW=4  RFL="${rf_levels[$idxB]}" TYPE="normal" FOLDER="${directory}" PR="${pruning_rates[$idxA]}" EXPERIMENT=1 FFCV_TRAIN="${ffcv_train}" FFCV_VAL="${ffcv_val}"

qsub -l h_rt=3:00:00 -l coproc_v100=1  -N "${model}_${rf_levels[$idxB]}_${dataset}_${pruning_rates[$idxA]}_pruning_ffcv" run_pruning_arc.sh FFCV="${ffcv}" NAME="${name}" MODEL="${model}" DATASET="${dataset}"  NUMW=4  RFL="${rf_levels[$idxB]}" TYPE="normal" FOLDER="${directory}" PR="${pruning_rates[$idxA]}" EXPERIMENT=1 FFCV_TRAIN="${ffcv_train}" FFCV_VAL="${ffcv_val}"

# slurm_pruning_run.sh FFCV="${ffcv}" NAME="${name}" MODEL="${model}" DATASET="${dataset}"  NUMW=4  RFL="${rf_levels[$idxB]}" TYPE="normal" FOLDER="${directory}" PR="${pruning_rates[$idxA]}" EXPERIMENT=1 FFCV_TRAIN="${ffcv_train}" FFCV_VAL="${ffcv_val}"

#./slurm_pruning_run.sh FFCV=0 NAME=recording_200_no_ffcv MODEL=resnet_small DATASET=small_imagenet NUMW=4 RFL=10 TYPE=normal FOLDER=$HOME/checkpoints PR=0.6 EXPERIMENT=1
else
  echo "Don't use FFCV"

#sbatch --nodes=1 --time=03:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk  --error="${model}_${rf_levels[$idxB]}_${dataset}_${pruning_rates[$idxA]}_pruning_no_ffcv.err" --output="${model}_${rf_levels[$idxB]}_${dataset}_${pruning_rates[$idxA]}_pruning_no_ffcv.out" --job-name="${model}_${rf_levels[$idxB]}_${dataset}_${pruning_rates[$idxA]}_pruning_no_ffcv" slurm_pruning_run.sh FFCV="${ffcv}" NAME="${name}" MODEL="${model}" DATASET="${dataset}"  NUMW=4  RFL="${rf_levels[$idxB]}" TYPE="normal" FOLDER="${directory}" PR="${pruning_rates[$idxA]}" EXPERIMENT=1

qsub -l h_rt=3:00:00 -l coproc_v100=1  -N "${model}_${rf_levels[$idxB]}_${dataset}_${pruning_rates[$idxA]}_pruning_ffcv" run_pruning_arc.sh FFCV="${ffcv}" NAME="${name}" MODEL="${model}" DATASET="${dataset}"  NUMW=4  RFL="${rf_levels[$idxB]}" TYPE="normal" FOLDER="${directory}" PR="${pruning_rates[$idxA]}" EXPERIMENT=1
#   slurm_run.sh  "${model}" "${dataset}" 4  "${rf_levels[$idxB]}" "normal" "${directory}" "${pruning_rates[$idxA]}" 1
#slurm_pruning_run.sh FFCV="${ffcv}" NAME="${name}" MODEL="${model}" DATASET="${dataset}"  NUMW=4  RFL="${rf_levels[$idxB]}" TYPE="normal" FOLDER="${directory}" PR="${pruning_rates[$idxA]}" EXPERIMENT=1
  fi
#echo "${model}_${rf_levels[$idxB]}_${dataset}_${pruning_rates[$idxA]}_pruning in directory ${directory}"
done
done
}
run_pruning "resnet_small" "small_imagenet" "${HOME}/checkpoints" "recording_200_no_ffcv" 0 "no_set" "no_set" 0
