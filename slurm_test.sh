#!/bin/bash
# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=00:30:00

# set name of job
#SBATCH --job-name=RF_Test

#SBATCH --error=RF_test.err

#SBATCH --output=RF_test.output

# set partition (devel, small, big)
#SBATCH --partition=small

# set number of GPUs
#S BATCH --gres=gpu:1

# mail alert at start, end and abortion of execution
#S BATCH --mail-type=NONE

# send mail to this address
#S BATCH --mail-user=sclaam@leeds.ac.uk
python hao_models_import.py