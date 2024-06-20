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


qsub -t 1-4 -l h_rt=24:00:00 -l coproc_v100=1  -N "training_Level_5_resnet18_small_imagenet" run.sh "resnet_small" "small_imagenet" 4 5 "normal" 200 "recording_200" 1 1 "/nobackup/sclaam/checkpoints"
qsub -t 1-4 -l h_rt=24:00:00 -l coproc_v100=1  -N "training_Level_7_resnet18_small_imagenet" run.sh "resnet_small" "small_imagenet" 4 7 "normal" 200 "recording_200" 1 1 "/nobackup/sclaam/checkpoints"
qsub -t 1-4 -l h_rt=24:00:00 -l coproc_v100=1  -N "training_Level_8_resnet18_small_imagenet" run.sh "resnet_small" "small_imagenet" 4 8  "normal" 200 "recording_200" 1 1 "/nobackup/sclaam/checkpoints"
qsub -t 1-4 -l h_rt=24:00:00 -l coproc_v100=1  -N "training_Level_9_resnet18_small_imagenet" run.sh "resnet_small" "small_imagenet" 4 9  "normal" 200 "recording_200" 1 1 "/nobackup/sclaam/checkpoints"
#"resnet_small" "small_imagenet" 4 10 "normal" 200 "recording_200_no_ffcv" 1 1
