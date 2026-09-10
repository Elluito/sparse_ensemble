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
#SBATCH --mem-per-cpu=4GB

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
echo "DATA_FOLDER= $DATA_FOLDER"
echo "SAVE_FOLDER= $SAVE_FOLDER"
echo "INPUT_RES= $INPUT_RES"

export LD_LIBRARY_PATH=""
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/users/$USER/.conda/envs/work/lib"
export PYTHONPATH="/users/$USER/.conda/envs/work/lib/python3.9/site-packages"

#############################################################
#     One-shot pruning (evaluation only, no fine-tuning) of the
#     learning-rate-sweep checkpoints in FOLDER matching
#     MODEL/DATASET/RFL/NAME, at pruning_rate=PR (prune_models.py,
#     experiment 1 -- globs every seed found and records dense-vs-
#     pruned accuracy).
#
#     Then, on the same checkpoints, saturation_calculation.py prunes
#     again on the fly (its own --pruning_rate branch) and computes
#     saturation of the pruned model.
#
#     Both write their csv output into SAVE_FOLDER.
#############################################################

python prune_models.py --experiment 1 --name "${NAME}" --model "${MODEL}" --dataset "${DATASET}" \
  --num_workers "${NUMW}" --RF_level "${RFL}" --type "${TYPE}" --folder "${FOLDER}" \
  --pruning_rate "${PR}" --data_folder "${DATA_FOLDER}" --save_folder "${SAVE_FOLDER}" \
  --input_resolution "${INPUT_RES}" --batch_size 128 --record 1

python saturation_calculation.py --name "${NAME}" --model "${MODEL}" --dataset "${DATASET}" \
  --num_workers "${NUMW}" --RF_level "${RFL}" --type "${TYPE}" --folder "${FOLDER}" \
  --pruning_rate "${PR}" --data_folder "${DATA_FOLDER}" --save_folder "${SAVE_FOLDER}" \
  --input_resolution "${INPUT_RES}" --batch_size 128
