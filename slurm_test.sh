#!/bin/bash
# set the number of nodes
#S BATCH --nodes=1

# set max wallclock time
#S BATCH --time=00:05:00

# set name of job
#S BATCH --job-name=Import_Test

#S BATCH --error=import_test.err

#S BATCH --output=import_test.output

# set partition (devel, small, big)
#S BATCH --partition=small

# set number of GPUs
#S BATCH --gres=gpu:1

# mail alert at start, end and abortion of execution
#S BATCH --mail-type=NONE

# send mail to this address
#S BATCH --mail-user=sclaam@leeds.ac.uk
