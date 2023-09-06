#!/bin/bash -l
# Submission script for serial Python job
# Luis Alfredo Avendano  2021-10-22

# Run from the current directory and with current environment
#$ -cwd -V

# Ask for some time (hh:mm:ss max of 48:00:00)
#$ -l h_rt=03:00:00

# ASk for some GPU
#$ -l coproc_p100=1

# Ask for some memory (by default, 1G, without a request)
#$ -l h_vmem=16G
# -pe smp 3
# Send emails when job starts and ends
#$ -m e
#export OMP_NUM_THREADS=3
#export MKL_NUM_THREADS=3
#export OPENBLAS_NUM_THREADS=1
# Now run the job
#module load intel openmpi
#module add anaconda
#module add cuda/11.1.1
#conda activate work2
#which python
#unset GOMP_CPU_AFFINITY
#nvcc --version
#python main.py $1 $2 $3 $4 $5 $6
#&& python main.py && python main.py
#
#CUDA_VISIBLE_DEVICES="0" accelerate launch --num_processes=1 --mixed_precision=fp16 main.py -exp $1 -bs 128 --sigma $2 --pruner $3 --architecture $4 --dataset $5 --pruning_rate $6 --modeltype $7 --epochs $8

models=("resnet18" "resnet50")
#Datasets
#datasets=("cifar10" "cifar100")
types=("alternative" "hub")
seeds=("1" "2")


for model in ${models[@]}; do
# For over the models
for type in ${types[@]};do
for seed in ${seeds[@]};do
#for f in ${functions[@]};do
# For over  datasets
if [ "${model}" = "resnet50" ] && [  "${seed}" = "2" ] & [  "${type}" = "normal" ] ; then
 continue
fi

#python main.py -exp 18 -bs 128 --sigma $2 --pruner $3 --architecture $4 --dataset $5 --pruning_rate $6 --modeltype $7 --epochs $8
qsub -N "saving_features_${model}_${type}_${seed}" run.sh  "${seed}" "0.0" "global" "${model}" "cifar10" "0.9" "${type}" "1" "nsga" "2"
#./run.sh  19 "0.0" "global" "${model}" "${dataset}" "0.9" "alternative" "1" "${sampler}" "2"
#./run.sh  1 "0.0" "global" "resnet18" "cifar10" "0.9" "alternative" "1" "nsga" "2"
#done
done
done

done
