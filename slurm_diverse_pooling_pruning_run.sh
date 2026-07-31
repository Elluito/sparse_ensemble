#!/bin/bash
# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=00:09:00

# set name of job
#SBATCH --job-name=pytorch_test

#SBATCH --error=pytorch_test.err

#SBATCH --output=pytorch_test.output

# set partition (devel, small, big)

#SBATCH --partition=small

# set number of GPUs
#SBATCH --gres=gpu:1

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=sclaam@leeds.ac.uk
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

module load miniforge/
conda activate work

for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

# use here your expected variables
echo "FFCV = $FFCV"
echo "NAME = $NAME"
echo "MODEL = $MODEL"
echo "DATASET= $DATASET"
echo "NUMW= $NUMW"
echo "RFL=$RFL"
echo "TYPE= $TYPE"
echo "FOLDER= $FOLDER"
echo "PR= $PR"
echo "EXPERIMENT= $EXPERIMENT"
echo "SOLUTION= $SOLUTION"
echo "FFCV_TRAIN = $FFCV_TRAIN "
echo "FFCV_VAL = $FFCV_VAL "
# Library variables
export LD_LIBRARY_PATH=""
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/users/sclaam/.conda/envs/work/lib"
export PYTHONPATH="/users/sclaam/.conda/envs/work/lib/python3.9/site-packages"
#############################################################
#     Prune (experiment 1) / Prune + fine-tune 10 epochs (experiment 2)
#############################################################

if [[ $FFCV -gt 0 ]]; then
echo "use FFCV!"

    python prune_models.py --ffcv --name "${NAME}" --model "${MODEL}" --dataset "${DATASET}" --num_workers "${NUMW}" --RF_level "${RFL}" --type "${TYPE}" --folder "${FOLDER}" --pruning_rate "${PR}" --experiment "${EXPERIMENT}" --solution "${SOLUTION}" --ffcv_train "${FFCV_TRAIN}" --ffcv_val "${FFCV_VAL}" --data_folder "${DATA_FOLDER}" --save_folder "${SAVE_FOLDER}" --input_resolution "${INPUT_RES}" --resize "${RESIZE}" --epochs 10 --record 1

else

    echo "Don't use FFCV!"

    python prune_models.py --name "${NAME}" --model "${MODEL}" --dataset "${DATASET}" --num_workers "${NUMW}" --RF_level "${RFL}" --type "${TYPE}" --folder "${FOLDER}" --pruning_rate "${PR}" --experiment "${EXPERIMENT}" --solution "${SOLUTION}" --data_folder "${DATA_FOLDER}" --save_folder "${SAVE_FOLDER}" --input_resolution "${INPUT_RES}" --resize "${RESIZE}" --epochs 10 --record 1

fi
