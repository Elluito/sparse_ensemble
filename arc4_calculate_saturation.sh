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
#$ -l h_vmem=18G
# -t 2-5
# -pe smp 3
# Send emails when job starts and ends
#$ -m be








conda activate work2


unset GOMP_CPU_AFFINITY
unset KMP_AFFINITY



if [ "${ffcv}" -gt 0 ]
  then

python prune_models.py --ffcv --name "${name}" --model "${model}" --dataset "${dataset}" --num_workers 0 --RF_level "${rf_level}" --type "normal" --folder "${directory}" --pruning_rate "${pruning_rate}"  --experiment 6 --data_folder "${data_folder}" --save_folder "${save_folder}" --input_resolution "${resolution}"  --resize "${resize}"
else

python prune_models.py --name "${name}" --model "${model}" --dataset "${dataset}" --num_workers 0 --RF_level "${rf_level}" --type "normal" --folder "${directory}" --pruning_rate "${pruning_rate}"  --experiment 6 --data_folder "${data_folder}" --save_folder "${save_folder}" --input_resolution "${resolution}"  --resize "${resize}"

  fi