#!/bin/bash
# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=05:30:00

# set name of job
#SBATCH --job-name=ffcv_statistics_small_imagenet

#SBATCH --error=ffcv_statistics_small_imagenet.err

#SBATCH --output=ffcv_statistics_small_imagenet.output

# set partition (devel, small, big)

#SBATCH --partition=small

# set number of GPUs

#SBATCH --gres=gpu:1

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=sclaam@leeds.ac.uk
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
export LD_LIBRARY_PATH=""
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/jmain02/home/J2AD014/mtc03/lla98-mtc03/.conda/envs/ffcv/lib"
export PYTHONPATH="/jmain02/home/J2AD014/mtc03/lla98-mtc03/.conda/envs/ffcv/lib/python3.9/site-packages"
which python
python comparison_small_imagenet.py
#python3.9 ffcv_loaders.py