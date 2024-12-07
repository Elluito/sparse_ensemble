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








conda activate work

which python
unset GOMP_CPU_AFFINITY
unset KMP_AFFINITY




python extraction_external.py -f $1 -m $2 -t $3 -s $4 -d $5 --RF_level $6 --input_resolution $7 --num_workers $8  --batch_size $9 --save_path "${10}" --name "${11}" --latent_folder "${12}" --downsampling "${13}" --experiment 2 --adjust_bn "${14}" --pruning_rate "${15}"
