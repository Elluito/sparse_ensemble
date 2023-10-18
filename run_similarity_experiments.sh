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

#models=("resnet50")
##Datasets
##datasets=("cifar10" "cifar100")
#types=("alternative")
#seeds=("2")
#
#
#for model in ${models[@]}; do
## For over the models
#for type in ${types[@]};do
#for seed in ${seeds[@]};do
#for f in ${functions[@]};do
# For over  datasets
#if [ "${model}" = "resnet50" ] && [  "${seed}" = "2" ] & [  "${type}" = "normal" ] ; then
# continue
#fi




######## FOR RESNET50       ###################################################################

# Level 0
rf_level0_s1="trained_models/cifar10/resnet50_cifar10.pth"
name_rf_level0_s1="_seed_1_rf_level_0"


rf_level0_s2="trained_models/cifar10/resnet50_normal_seed_2_tst_acc_95.65.pth"
name_rf_level0_s2="_seed_2_rf_level_0"

# level 1
rf_level1_s1="trained_models/cifar10/resnet50_normal_cifar10_seed_1_rf_level_1_95.26.pth"
name_rf_level1_s1="_seed_1_rf_level_1"
rf_level1_s2="trained_models/cifar10/resnet50_normal_cifar10_seed_2_rf_level_1_94.93.pth"
name_rf_level1_s2="_seed_2_rf_level_1"

# level 2
rf_level2_s1="trained_models/cifar10/resnet50_normal_cifar10_seed_1_rf_level_2_94.07.pth"
name_rf_level2_s1="_seed_1_rf_level_2"

rf_level2_s2="trained_models/cifar10/resnet50_normal_cifar10_seed_2_rf_level_2_94.03.pth"
name_rf_level2_s2="_seed_2_rf_level_2"
#Level 3

rf_level3_s1="trained_models/cifar10/resnet50_normal_cifar10_seed_1_rf_level_3_92.38.pth"
name_rf_level3_s1="_seed_1_rf_level_3"

rf_level3_s2="trained_models/cifar10/resnet50_normal_cifar10_seed_2_rf_level_3_92.25.pth"
name_rf_level3_s2="_seed_2_rf_level_3"


#Level 4
rf_level4_s1="trained_models/cifar10/resnet50_normal_cifar10_seed_1_rf_level_4_90.66.pth"
name_rf_level4_s1="_seed_1_rf_level_4"
rf_level4_s2="trained_models/cifar10/resnet50_normal_cifar10_seed_2_rf_level_4_90.8.pth"
name_rf_level4_s2="_seed_2_rf_level_4"

# Pytorch implementation

rf_level_p_s1="trained_models/cifar10/resnet50_official_cifar10_seed_1_test_acc_90.31.pth"
name_rf_level_p_s1="_seed_1_rf_level_p"

rf_level_p_s2="trained_models/cifar10/resnet50_official_cifar10_seed_2_test_acc_89.93.pth"
name_rf_level_p_s2="_seed_2_rf_level_p"

rf_level_p_s3="trained_models/cifar10/resnet50_pytorch_cifar10_seed_3_test_acc_89.33.pth"
name_rf_level_p_s3="_seed_3_rf_level_p"





#python main.py -exp 18 -bs 128 --sigma $2 --pruner $3 --architecture $4 --dataset $5 --pruning_rate $6 --modeltype $7 --epochs $8

qsub -N "saving_features_resnet50_rf_levelp_s3_pytorch_no_train" run.sh  "resnet50" "filer" "pytorch_no_train_1" 0 hub
qsub -N "saving_features_resnet50_rf_levelp_s3_pytorch_no_train" run.sh  "resnet50" "filer" "pytorch_no_train_2" 0 hub
qsub -N "saving_features_resnet50_rf_level0_s3_pytorch_no_train" run.sh  "resnet50" "filer" "custom_no_train_1" 0 alternative
qsub -N "saving_features_resnet50_rf_level0_s3_pytorch_no_train" run.sh  "resnet50" "filer" "custom_no_train_2" 0 alternative
#./run.sh  "resnet50" foreing_trained_models/cifar10/resnet50_pytorch_cifar10_seed_3_test_acc_89.33.pth "${name_rf_level_p_s3}" 0 hub
#qsub -N "saving_features_resnet50_rf_level1_s1" run.sh  "resnet50" "${rf_level1_s1}" "${name_rf_level1_s1}" 1 alternative
#qsub -N "saving_features_resnet50_rf_level1_s2" run.sh  "resnet50" "${rf_level1_s2}" "${name_rf_level1_s2}" 1 alternative
#qsub -N "saving_features_resnet50_rf_level3_s1" run.sh  "resnet50" "${rf_level3_s1}" "${name_rf_level3_s1}" 3 alternative
#qsub -N "saving_features_resnet50_rf_level3_s2" run.sh  "resnet50" "${rf_level3_s2}" "${name_rf_level3_s2}" 3 alternative
#qsub -N "saving_features_resnet50_rf_level2_s1" run.sh  "resnet50" "${rf_level2_s1}" "${name_rf_level2_s1}" 2 alternative
#qsub -N "saving_features_resnet50_rf_level2_s2" run.sh  "resnet50" "${rf_level2_s2}" "${name_rf_level2_s2}" 2 alternative
#qsub -N "saving_features_resnet50_rf_level4_s1" run.sh  "resnet50" "${rf_level4_s1}" "${name_rf_level4_s1}" 4 alternative
#qsub -N "saving_features_resnet50_rf_level4_s2" run.sh  "resnet50" "${rf_level4_s2}" "${name_rf_level4_s2}" 4 alternative

#
#files=($name_rf_level1_s1 $name_rf_level1_s2 $name_rf_level2_s1 $name_rf_level2_s2 $name_rf_level3_s1 $name_rf_level3_s2 $name_rf_level4_s1 $name_rf_level4_s2)
#max=${#files[@]}                                  # Take the length of that array
#for ((idxA=0; idxA<max; idxA++)); do              # iterate idxA from 0 to length
#  for ((idxB=idxA+1; idxB<max; idxB++)); do         # iterate idxB from idxA to length
#    echo "A: ${files[$idxA]}; B: ${files[$idxB]}" # Do whatever you're here for.
#    qsub -N "similarity_${files[$idxA]}_${files[$idxB]}" run.sh  "resnet50" "${files[$idxA]}" "${files[$idxB]}" "alternative" "alternative" "npy" "npy"
#  done
#done

#qsub -N "similarity_tird_seed_pytorch_1" run.sh  "resnet50" "_seed_1" "${name_rf_level_p_s3}" "hub" "hub" "txt" "npy"
#qsub -N "similarity_tird_seed_pytorch_2" run.sh  "resnet50" "_seed_2" "${name_rf_level_p_s3}" "hub" "hub" "txt" "npy"
#
#qsub -N "similarity_tird_seed_pytorch_1_k80" run.sh  "resnet50"  "${name_rf_level_p_s3}" "_seed_2" "${rf_level_p_s3}"
#./run.sh  "resnet50"  "${name_rf_level_p_s3}" "_seed_2" "${rf_level_p_s3}"



### Smoothness#######################

#files_names=($name_rf_level0_s1 $name_rf_level1_s1  $name_rf_level2_s1 $name_rf_level3_s1 $name_rf_level4_s1 $name_rf_level_p_s1)
#files=($rf_level0_s1 $rf_level1_s1  $rf_level2_s1 $rf_level3_s1 $rf_level4_s1 $rf_level_p_s1)
#files_level=(0 1 2 3 4 0)
#files_type=("normal" "normal" "normal" "normal" "normal" "pytorch")
#max=${#files[@]}                                  # Take the length of that array
#for ((idxA=0; idxA<max; idxA++)); do              # iterate idxA from 0 to length
#qsub -N "smoothness_landscape${files_names[$idxA]}" run.sh  "resnet50" "cifar10" "${files_level[$idxA]}" "${files_type[$idxA]}" "${files_names[$idxA]}" "${files[$idxA]}"
#done

#./run.sh  19 "0.0" "global" "${model}" "${dataset}" "0.9" "alternative" "1" "${sampler}" "2"
#./run.sh  1 "0.0" "global" "resnet18" "cifar10" "0.9" "alternative" "1" "nsga" "2"
##done
#done
#done
#
#done
