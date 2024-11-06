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
#$ -l h_vmem=16G
# -t 2-5
# -pe smp 3
# Send emails when job starts and ends
#$ -m be


#python3.9 -c "import torch;device = 'cuda' if torch.cuda.is_available() else 'cpu';print(device);print('Cuda version with torch: {}'.format(torch.version.cuda))"
#export LD_LIBRARY_PATH=""
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/jmain02/home/J2AD014/mtc03/lla98-mtc03/.conda/envs/ffcv/lib"
#export PYTHONPATH="/jmain02/home/J2AD014/mtc03/lla98-mtc03/.conda/envs/ffcv/lib/python3.9/site-packages"


conda activate ffcv
which python
unset GOMP_CPU_AFFINITY
unset KMP_AFFINITY
python3.9 -c "import os; print(os.environ)"

python3.9 extraction_external.py -f $1 -m $2 -t $3 -s $4 -d $5 --RF_level $6 --input_resolution $7 --num_workers $8  --batch_size $9 --save_path "${10}" --name "${11}"
