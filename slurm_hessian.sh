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

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

# send mail to this address
#SBATCH --mail-user=sclaam@leeds.ac.uk
#module load pytorch

which python

for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

#python train_CIFAR10.py --model $1 --dataset $2 --num_workers $3 --RF_level $4 --type $5

#python Second_order_Receptive_field.py --experiment 2 --optimiser "kfac"
#
#python Second_order_Receptive_field.py --experiment 2 --optimiser "sam"

#  if [ $6 -eq 1 ]
#  then
#    # KFac
#python Second_order_Receptive_field.py --lr "0.01" --momentum "0.5" --grad_clip "1" --save 1 --experiment 1 --epochs 100 --batch_size 32 --use_scheduler 1 --use_scheduler_batch 0 --num_workers 4 --optimiser "kfac" --record 1 -dt $1 --model $2 --RF_level $3 --type $4 --name $5 --save_folder "$HOME/checkpoints"
#
#  fi
#
#  if [ $6 -eq 2 ]
#    # ASAM
#  then
#python Second_order_Receptive_field.py --lr "0.1" --momentum "0.7" --grad_clip "1" --save 1 --experiment 1 --epochs 100 --batch_size 128 --use_scheduler 1 --use_scheduler_batch 0 --num_workers 4 --optimiser "sam" --record 1 -dt $1 --model $2 --RF_level $3 --type $4 --name $5 --save_folder "$HOME/checkpoints"
#  fi
#
#python -c "import torch;device = 'cuda' if torch.cuda.is_available() else 'cpu';print(device);print('Cuda version with torch: {}'.format(torch.version.cuda))"
#python -c "import os; print(os.environ)"
#printf "Start Test \n"
#python test_backwards.py
#printf "End Test \n"
#module load pytorch
#echo "After loading the pytorch module"
#which python
#python -c "import torch;device = 'cuda' if torch.cuda.is_available() else 'cpu';print(device);print('Cuda version with torch: {}'.format(torch.version.cuda))"
#python -c "import os; print(os.environ)"
#printf "Start Test \n"
#python test_backwards.py
#printf "End Test \n"
##echo "============ 2 workers ============================"
##python hao_models_pruning_test.py --workers 2
#echo "============ 4 workers ============================"
#python hao_models_pruning_test.py --workers 4
#echo "============ 8 workers ============================"
#python hao_models_pruning_test.py --workers 2 --experiment 2 --model $1
#echo "============ 16 workers ============================"
#python hao_models_pruning_test.py --workers 16
#echo "============ 32 workers ============================"
#python hao_models_pruning_test.py --workers 32
#unset GOMP_CPU_AFFINITY
#unset KMP_AFFINITY
python -c "import os; print(os.environ)"
python -c "import torch;device = 'cuda' if torch.cuda.is_available() else 'cpu';print(device);print('Cuda version with torch: {}'.format(torch.version.cuda))"

  if [[ $FFCV -gt 0 ]]; then
#    export LD_LIBRARY_PATH="/jmain02/home/J2AD014/mtc03/lla98-mtc03/.conda/envs/ffcv/lib"
#    echo "CPUs allocated: $SLURM_JOB_CPUS_PER_NODE"
#    python -c "import os; print(os.environ)"
#    which python
#     recording_200_ffcv
echo "use FFCV!"
export LD_LIBRARY_PATH=""
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/jmain02/home/J2AD014/mtc03/lla98-mtc03/.conda/envs/ffcv/lib"

#    python prune_models.py --name "${NAME}" --ffcv --model $1 --dataset $2 --num_workers $3 --RF_level $4 --type $5 --folder $6 --pruning_rate $7 --experiment $8

 if [[ $BATCH_ONLY -gt 0 ]]; then

   python calculate_hessian.py --ffcv  --name "${NAME}" --model "${MODEL}" --dataset "${DATASET}" --num_workers "${NUMW}" --RF_level "${RFL}" --type "${TYPE}" --folder "${FOLDER}"   --ffcv_train "${FFCV_TRAIN}" --ffcv_val "${FFCV_VAL}" --n_eigenvalues "${N_EIGEN}" --n_buffer "${N_BUFFER}" --solution "${SOLUTION}"  --batch_only  --eval_size "${EVAL_SIZE}" --method "${METHOD}" --data_folder "${DATA_FOLDER}"

else
   python calculate_hessian.py --ffcv  --name "${NAME}" --model "${MODEL}" --dataset "${DATASET}" --num_workers "${NUMW}" --RF_level "${RFL}" --type "${TYPE}" --folder "${FOLDER}"   --ffcv_train "${FFCV_TRAIN}" --ffcv_val "${FFCV_VAL}" --n_eigenvalues "${N_EIGEN}" --n_buffer "${N_BUFFER}" --solution "${SOLUTION}"   --eval_size "${EVAL_SIZE}" --method "${METHOD}" --data_folder "${DATA_FOLDER}"

  fi
else

#    echo "CPUs allocated: $SLURM_JOB_CPUS_PER_NODE"
#    python -c "import os; print(os.environ)"
#    which python
    echo "Don't use FFCV!"
#./slurm_pruning_run.sh FFCV=0 NAME=recording_200_no_ffcv MODEL=resnet_small DATASET=small_imagenet NUMW=4 RFL=10 TYPE=normal FOLDER=$HOME/checkpoints PR=0.6 EXPERIMENT=1

 if [[ $BATCH_ONLY -gt 0 ]]; then
    python calculate_hessian.py --name "${NAME}" --model "${MODEL}" --dataset "${DATASET}" --num_workers "${NUMW}" --RF_level "${RFL}" --type "${TYPE}" --folder "${FOLDER}" --n_eigenvalues "${N_EIGEN}" --n_buffer "${N_BUFFER}" --solution "${SOLUTION}"  --batch_only --eval_size "${EVAL_SIZE}" --method "${METHOD}" --data_folder "${DATA_FOLDER}"
    else

    python calculate_hessian.py --name "${NAME}" --model "${MODEL}" --dataset "${DATASET}" --num_workers "${NUMW}" --RF_level "${RFL}" --type "${TYPE}" --folder "${FOLDER}" --n_eigenvalues "${N_EIGEN}" --n_buffer "${N_BUFFER}" --solution "${SOLUTION}"  --eval_size "${EVAL_SIZE}" --method "${METHOD}" --data_folder "${DATA_FOLDER}"
      fi
  fi
