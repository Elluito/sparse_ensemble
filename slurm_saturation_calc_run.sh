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

# ####SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

# send mail to this address
#SBATCH --mail-user=sclaam@leeds.ac.uk
#module load pytorch


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
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/users/sclaam/.conda/envs/work/lib"
export PYTHONPATH="/users/sclaam/.conda/envs/work/lib/python3.9/site-packages"
#export PYTHONPATH="/jmain02/home/J2AD014/mtc03/lla98-mtc03/.conda/envs/ffcv/lib/python3.9/site-packages"

#unset GOMP_CPU_AFFINITY KMP_AFFINITY






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
echo "FFCV_TRAIN = $FFCV_TRAIN "
echo "FFCV_VAL = $FFCV_VAL "
echo "SAVE_FOLDER= $SAVE_FOLDER"

#    python prune_models.py --name "${NAME}" --ffcv --model $1 --dataset $2 --num_workers $3 --RF_level $4 --type $5 --folder $6 --pruning_rate $7 --experiment $8

echo "EXTRA_VALUES = $EXTRA_VALUES"

#############################################################
#     One shot with specific pruning rate results
#############################################################

  if [[ $FFCV -gt 0 ]]; then
echo "use FFCV!"

    python3.9 saturation_calculation.py --ffcv --name "${NAME}" --model "${MODEL}" --dataset "${DATASET}" --num_workers "${NUMW}" --RF_level "${RFL}" --type "${TYPE}" --folder "${FOLDER}" --pruning_rate "${PR}" --ffcv_train "${FFCV_TRAIN}" --ffcv_val "${FFCV_VAL}" --data_folder "${DATA_FOLDER}" --save_folder "${SAVE_FOLDER}"
else


    echo "Don't use FFCV!"

    python3.9 saturation_calculation.py --name "${NAME}" --model "${MODEL}" --dataset "${DATASET}" --num_workers "${NUMW}" --RF_level "${RFL}" --type "${TYPE}" --folder "${FOLDER}" --pruning_rate "${PR}" --data_folder "${DATA_FOLDER}" --save_folder "${SAVE_FOLDER}"
  fi
