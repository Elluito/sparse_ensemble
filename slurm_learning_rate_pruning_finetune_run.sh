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
echo "PR= $PR"
echo "SOLUTION= $SOLUTION"
echo "DATA_FOLDER= $DATA_FOLDER"
echo "SAVE_FOLDER= $SAVE_FOLDER"
echo "INPUT_RES= $INPUT_RES"
echo "RESIZE= $RESIZE"

export LD_LIBRARY_PATH=""
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/users/${USER}/.conda/envs/work/lib"
export PYTHONPATH="/users/${USER}/.conda/envs/work/lib/python3.9/site-packages"

#############################################################
#     Prune (experiment 1) / Prune + fine-tune 10 epochs (experiment 2)
#     of a single named learning-rate-sweep checkpoint (SOLUTION), at
#     pruning_rate=PR. Output (fine-tuned pruned checkpoint + accuracy log)
#     lands under FOLDER/pruned/PR (prune_models.py's own convention),
#     same as slurm_diverse_pooling_pruning_run.sh.
#############################################################

python prune_models.py --name "${NAME}" --model "${MODEL}" --dataset "${DATASET}" --num_workers "${NUMW}" \
  --RF_level "${RFL}" --type "${TYPE}" --folder "${FOLDER}" --pruning_rate "${PR}" --experiment 2 \
  --solution "${SOLUTION}" --data_folder "${DATA_FOLDER}" --save_folder "${SAVE_FOLDER}" \
  --input_resolution "${INPUT_RES}" --resize "${RESIZE}" --epochs 10 --record 1
