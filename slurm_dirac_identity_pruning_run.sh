#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --job-name=dirac_identity_pruning
#SBATCH --error=dirac_identity_pruning.err
#SBATCH --output=dirac_identity_pruning.output
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=l.avendanomunoz@leeds.ac.uk

module load miniforge/
conda activate work

for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)
   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"
   export "$KEY"="$VALUE"
done

echo "RFL=$RFL"
echo "FOLDER=$FOLDER"
echo "SAVE_FOLDER=$SAVE_FOLDER"
echo "PRS=$PRS"

export LD_LIBRARY_PATH=""
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/users/$USER/.conda/envs/work/lib"
export PYTHONPATH="/users/$USER/.conda/envs/work/lib/python3.9/site-packages"

#############################################################
#     Identity (Dirac centre-tap) vs learned-weight survival under one-shot global magnitude pruning, for every
#     DiracNet-50 lr-sweep checkpoint of RF level RFL (all 17 lrs, every seed), at pruning rates PRS (space
#     separated). Weight magnitudes only: no data, no GPU. --dirac_width 4 must match training.
#############################################################

python measure_dirac_identity_pruning.py --folder "${FOLDER}" --save_folder "${SAVE_FOLDER}" \
  --rf_levels "${RFL}" --pruning_rates ${PRS} --dirac_width 4
