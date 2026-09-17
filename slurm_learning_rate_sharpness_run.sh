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

#SBATCH --partition=gpu

# set number of GPUs
#SBATCH --gres=gpu:1

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

# send mail to this address
#SBATCH --mail-user=sclaam@leeds.ac.uk

module load miniforge/
conda activate work

for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

echo "NAME = $NAME"
echo "MODEL = $MODEL"
echo "DATASET= $DATASET"
echo "NUMW= $NUMW"
echo "RFL=$RFL"
echo "TYPE= $TYPE"
echo "FOLDER= $FOLDER"
echo "ALGO= $ALGO"
echo "DATA_FOLDER= $DATA_FOLDER"
echo "SAVE_FOLDER= $SAVE_FOLDER"

export LD_LIBRARY_PATH=""
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/users/${USER}/.conda/envs/work/lib"
export PYTHONPATH="/users/${USER}/.conda/envs/work/lib/python3.9/site-packages"

#############################################################
#     Sharpness (calculate_sharpness.py, algorithm=ALGO) of the
#     learning-rate-sweep dense checkpoints in FOLDER matching
#     MODEL/DATASET/RFL/NAME -- one row per seed found via glob,
#     written to "{SAVE_FOLDER}/{MODEL}_{RFL}_{DATASET}_{NAME}_sharpness_summary.csv"
#     (calculate_sharpness.py's own naming convention).
#############################################################

python calculate_sharpness.py --name "${NAME}" --model "${MODEL}" --dataset "${DATASET}" --num_workers "${NUMW}" \
  --RF_level "${RFL}" --type "${TYPE}" --folder "${FOLDER}" --algorithm "${ALGO}" \
  --data_folder "${DATA_FOLDER}" --save_folder "${SAVE_FOLDER}"
