#!/bin/bash
#$ -cwd -V
# -l h_rt=01:00:00
#$ -pe smp 8
# Send emails when job starts and ends
#$ -m be
#$ -l h_vmem=16G

#python3.9 -c "import os; print(os.environ)"
##python -c "import torch;device = 'cuda' if torch.cuda.is_available() else 'cpu';print(device);print('Cuda version with torch: {}'.format(torch.version.cuda))"
#
#export LD_LIBRARY_PATH=""
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/jmain02/home/J2AD014/mtc03/lla98-mtc03/.conda/envs/ffcv/lib"
#export PYTHONPATH="/jmain02/home/J2AD014/mtc03/lla98-mtc03/.conda/envs/ffcv/lib/python3.9/site-packages"
#
#
#python3.9 train_probes.py -m $1 -d $2 --RF_level $3 --input_resolution $4 -mp 4 --save_path $5 --name $6
#
#
#python3.9 -c "import os; print(os.environ)"
##python -c "import torch;device = 'cuda' if torch.cuda.is_available() else 'cpu';print(device);print('Cuda version with torch: {}'.format(torch.version.cuda))"
#
export LD_LIBRARY_PATH=""
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/nobackup/sclaam/.conda/envs/work/lib"
#export PYTHONPATH="/jmain02/home/J2AD014/mtc03/lla98-mtc03/.conda/envs/ffcv/lib/python3.9/site-packages"

unset GOMP_CPU_AFFINITY KMP_AFFINITY







python main.py -exp $1 -bs 128 --sigma $2 --pruner $3 --architecture $4 --dataset $5 --pruning_rate $6 --modeltype $7 --epochs $8 --name $9 -pop 10

#python main.py