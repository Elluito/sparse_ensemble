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

#qsub -N "saving_features_resnet50_rf_levelp_s3_pytorch_no_train" run.sh  "resnet50" "filer" "pytorch_no_train_1" 0 hub
#qsub -N "saving_features_resnet50_rf_levelp_s3_pytorch_no_train" run.sh  "resnet50" "filer" "pytorch_no_train_2" 0 hub
#qsub -N "saving_features_resnet50_rf_level0_s3_pytorch_no_train" run.sh  "resnet50" "filer" "custom_no_train_1" 0 alternative
#qsub -N "saving_features_resnet50_rf_level0_s3_pytorch_no_train" run.sh  "resnet50" "filer" "custom_no_train_2" 0 alternative

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

#qsub -N "similarity_no_train_1" run.sh  "resnet50" "pytorch_no_train_1" "pytorch_no_train_2" "hub" "hub" "npy" "npy"
#qsub -N "similarity_no_train_2" run.sh  "resnet50" "custom_no_train_1" "custom_no_train_2" "alternative" "alternative" "npy" "npy"
#
#qsub -N "similarity_tird_seed_pytorch_1_k80" run.sh  "resnet50"  "${name_rf_level_p_s3}" "_seed_2" "${rf_level_p_s3}"
#./run.sh  "resnet50"  "${name_rf_level_p_s3}" "_seed_2" "${rf_level_p_s3}"



###############################################################################################
#                                              Smoothness
###############################################################################################
#files_names=($name_rf_level0_s1 $name_rf_level1_s1  $name_rf_level2_s1 $name_rf_level3_s1 $name_rf_level4_s1 $name_rf_level_p_s1)
#files=($rf_level0_s1 $rf_level1_s1  $rf_level2_s1 $rf_level3_s1 $rf_level4_s1 $rf_level_p_s1)
#files_level=(0 1 2 3 4 0)
#files_type=("normal" "normal" "normal" "normal" "normal" "pytorch")
#max=${#files[@]}                                  # Take the length of that array
#for ((idxA=0; idxA<max; idxA++)); do              # iterate idxA from 0 to length
#qsub -N "loading_model_test_${files_names[$idxA]}" run.sh  "resnet50" "cifar10" "${files_level[$idxA]}" "${files_type[$idxA]}" "${files_names[$idxA]}" "${files[$idxA]}"
#echo "loading_model_test_${files_names[$idxA]}"
#./run.sh  "resnet50" "cifar10" "${files_level[$idxA]}" "${files_type[$idxA]}" "${files_names[$idxA]}" "${files[$idxA]}"
#done
#
#files_names=($name_rf_level0_s2 $name_rf_level1_s2  $name_rf_level2_s2 $name_rf_level3_s2 $name_rf_level4_s2 $name_rf_level_p_s2)
#files=($rf_level0_s2 $rf_level1_s2  $rf_level2_s2 $rf_level3_s2 $rf_level4_s2 $rf_level_p_s2)

#for ((idxA=0; idxA<max; idxA++)); do              # iterate idxA from 0 to length
#      qsub -N "${files_names[$idxA]}" run.sh  "resnet50" "cifar10" "${files_level[$idxA]}" "${files_type[$idxA]}" "${files_names[$idxA]}" "${files[$idxA]}"
#      echo "loading_model_test_${fles_names[$idxA]}"
#      ./run.sh  "resnet50" "cifar10" "${files_level[$idxA]}" "${files_type[$idxA]}" "${files_names[$idxA]}" "${files[$idxA]}"
#done

#models=("resnet50")
##Datasets
#datasets=("cifar10" "cifar100")
#types=("alternative")
#rf_levels=("2")
########################################################################################################################
#                   Training a model on a given dataset with a given receptive field
########################################################################################################################
#qsub -N "training_Level_0_vgg" run.sh "vgg19" "tiny_imagenet" 2 0 "normal" 300
#qsub -N "training_Level_1_vgg" run.sh "vgg19" "tiny_imagenet" 2 1 "normal" 300
#qsub -N "training_Level_2_vgg" run.sh "vgg19" "tiny_imagenet" 2 2 "normal" 300
#qsub -N "training_Level_3_vgg" run.sh "vgg19" "tiny_imagenet" 2 3 "normal" 300
#qsub -N "training_Level_4_vgg" run.sh "vgg19" "tiny_imagenet" 2 4 "normal" 300

#qsub -N "training_Level_0_rs" run.sh "resnet50" "tiny_imagenet" 2 0 "normal" 300
# -l coproc_p100=1
#qsub -l coproc_p100=1 -t 1-5 -N "TL_4_BS_64_rs" run.sh "resnet50" "tiny_imagenet" 8 4 "normal" 300 "bs_32"
#qsub -l coproc_p100=1 -t 1-5 -N "TL_2_BS_64_rs" run.sh "resnet50" "tiny_imagenet" 8 2 "normal" 300 "bs_32"
#qsub -l coproc_k80=1 -t 4-5 -N  "training_Level_1_rs.2" run.sh "resnet50" "tiny_imagenet" 8 1 "normal" 300
#qsub -N "training_Level_2_rs" run.sh "resnet50" "tiny_imagenet" 2 2 "normal" 300
#qsub -N "training_Level_3_rs" run.sh "resnet50" "tiny_imagenet" 2 3 "normal" 300
#qsub -N "training_Level_4_rs" run.sh "resnet50" "tiny_imagenet" 2 4 "normal" 300

########################################################################################################################


#     initial weights feature representation
#files_level=(0 0 0 1 1 1 2 2 2 3 3 3 4 4 4)
#file_seed=(3 4 5 3 4 5 3 4 5 3 4 5 3 4 5)
#
#max=${#files_level[@]}                                  # Take the length of that array
#for ((idxA=0; idxA<max; idxA++)); do              # iterate idxA from 0 to length
#  qsub -N "features_initial_weights_${files_level[$idxA]}_seed_${file_seed[$idxA]}" run.sh  "resnet50" "/nobackup/sclaam/checkpoints/resnet50_normal_cifar10_seed_${file_seed[$indxA]}_rf_level_${files_level[$indxA]}_initial_weights.pth" "_no_train_seed_${file_seed[$idxA]}_rf_level_${files_level[$idxA]}"  "${files_level[$idxA]}" "alternative"
#done
#

#levels_max=${#rf_levels[@]}                                  # Take the length of that array
#seeds_max=${#seeds[@]}                                  # Take the length of that array
#for ((idxA=0; idxA<levels_max; idxA++)); do              # iterate idxA from 0 to length
#for ((idxB=0; idxB<seeds_max; idxB++)); do              # iterate idxA from 0 to length
#for ((idxC=idxB+1; idxC<seeds_max; idxC++)); do              # iterate idxA from 0 to length
#
##echo "seed_${seeds[$idxB]}_VS_seed_${seeds[$idxC]}_level_${rf_levels[$idxA]}"
#qsub -N "similarity_level_${rf_levels[$idxA]}_seeds_${seeds[$idxB]}_${seeds[$idxC]}" run.sh  "resnet50" "_no_train_seed_${seeds[$idxB]}_rf_level_${rf_levels[$idxA]}" "_no_train_seed_${seeds[$idxC]}_rf_level_${rf_levels[$idxA]}" "alternative" "alternative" "npy" "npy"

#done
#done
#done
#########################################################
#             Transform seed name into normal number
#########################################################

########################################################################################################################
#                 Hessian spectra of solutions
########################################################################################################################
#
#model="resnet50"
#dataset="cifar10"
#init=true
#if [ $init ]; then
#    solution_string="initial_weights"
#
#else
#
#
#    solution_string="test_acc"
#fi
#
#
#directory=/nobackup/sclaam/checkpoints
#resnet50_normal_cifar10_seed_0_3_rf_level_1_initial_weights.pth
#level_1_seed0=($(ls $directory | grep -i "${model}.*${dataset}.*_seed_0_.*_level_1_${solution_string}.*"))
#echo $level_1_seed0
##level_1_seed1=($(ls $directory | grep -i "vgg19.*_seed_1_rf_level_1_test_acc.*"))
##echo $level_1_seed1
##
##level1_seeds=($level_1_seed0 $level_1_seed1)
##
##level_2_seed0=($(ls $directory | grep -i "vgg19.*_seed_0_rf_level_2_test_acc.*"))
#level_2_seed0=($(ls $directory | grep -i "${model}.*${dataset}.*_seed_0_.*_level_2_${solution_string}.*"))
#echo $level_2_seed0
##level_2_seed1=($(ls $directory | grep -i "vgg19.*_seed_1_rf_level_2_test_acc.*"))
##echo $level_2_seed1
##
##level2_seeds=($level_2_seed0 $level_2_seed1)
##
#level_3_seed0=($(ls $directory | grep -i "${model}.*${dataset}.*_seed_0_.*_level_3_${solution_string}.*"))
##level_3_seed0=($(ls $directory | grep -i "vgg19.*_seed_0_rf_level_3_test_acc.*"))
##
#echo $level_3_seed0
##level_3_seed1=($(ls $directory | grep -i "vgg19.*_seed_1_rf_level_3_test_acc.*"))
##echo $level_3_seed1
##
##level3_seeds=($level_3_seed0 $level_3_seed1)
##
#
##level_4_seed0=($(ls $directory | grep -i "vgg19.*_seed_0_rf_level_4_test_acc.*"))
#level_4_seed0=($(ls $directory | grep -i "${model}.*${dataset}.*_seed_0_.*_rf_level_4_${solution_string}.*"))
#echo $level_4_seed0
##level_4_seed1=($(ls $directory | grep -i "vgg19.*_seed_1_rf_level_4_test_acc.*"))
##echo $level_4_seed1
##level4_seeds=($level_4_seed0 $level_4_seed1)
#
#
#seeds="0"
#rf_levels=(1 2 3 4)
#levels_max=${#rf_levels[@]}                                  # Take the length of that array
#for ((idxA=0; idxA<levels_max; idxA++)); do              # iterate idxA from 0 to length
#
#
##levels_by_seed=(${level1_seeds[$idxB]} ${level2_seeds[$idxB]} ${level3_seeds[$idxB]} ${level4_seeds[$idxB]})
#
#levels_by_seed=(${level_1_seed0} ${level_2_seed0} ${level_3_seed0} ${level_4_seed0})
#echo "${directory}/${levels_by_seed[$idxA]}"
##echo "${levels_by_seed}"
#
#qsub -N "${model}_hessian_init_${dataset}_${rf_levels[$idxA]}" run.sh  "${model}" "${dataset}" "${rf_levels[$idxA]}" "normal" "seed_0_rf_level_${rf_levels[$idxA]}_init" "${directory}/${levels_by_seed[$idxA]}"
#
#done
#
########################################################################################################################
#                 Creating features loop and other not comparative experiments
########################################################################################################################

#seeds=(0 1)
#rf_levels=(1 2 3 4)
#levels_max=${#rf_levels[@]}                                  # Take the length of that array
#seeds_max=${#seeds[@]}                                  # Take the length of that array
#for ((idxA=0; idxA<levels_max; idxA++)); do              # iterate idxA from 0 to length
#for ((idxB=0; idxB<seeds_max; idxB++)); do              # iterate idxA from 0 to length
#
#
#levels_by_seed=(${level1_seeds[$idxB]} ${level2_seeds[$idxB]} ${level3_seeds[$idxB]} ${level4_seeds[$idxB]})
#
#
#
##temp=${levels_by_seed[$idxA]}
##echo $temp
##echo ${level1_seeds[$idxB]}
#echo "level ${rf_levels[$idxA]} seed ${seeds[$idxB]}"
#echo "solution ${levels_by_seed[$idxA]}"
#
#qsub -N "features_vgg19_${rf_levels[$idxA]}_seed_${seeds[$idxB]}" run.sh  "vgg19" "${directory}/${levels_by_seed[$idxA]}" "_seed_${seeds[$idxB]}_rf_level_${rf_levels[$idxA]}"  "${rf_levels[$idxA]}" "alternative"
#
##levels_by_seed=(${level1_seeds[0]} ${level2_seeds[0]} ${level3_seeds[0]} ${level4_seeds[0]})
#
##qsub -N "vgg19_pruning_summary_level_${rf_levels[$idxA]}" run.sh "vgg19" "cifar10" "2" "${rf_levels[$idxA]}" "normal" "${directory}"
##qsub -N "resnet50_pruning_summary_level_${rf_levels[$idxA]}" run.sh "resnet50" "cifar10" "2" "${rf_levels[$idxA]}" "normal" "${directory}"
#
#
##done
#done
#

########################################################################################################################
#                 Prune summary
########################################################################################################################






#directory=/nobackup/sclaam/checkpoints
#seeds=(0 1 2 3 4)
#rf_levels=(1 4)
#levels_max=${#rf_levels[@]}                                  # Take the length of that array
#number_of_elements_by_seed=${#[@]}
#for ((idxA=0; idxA<levels_max; idxA++)); do              # iterate idxA from 0 to length
#qsub -N "${model}_pruning_summary_level_${rf_levels[$idxA]}" run.sh "${model}" "${dataset}" "2" "${rf_levels[$idxA]}" "normal" "${directory}" "_32_bs"
#done

########################################################################################################################
#                 Prune and fine tune summary
########################################################################################################################
#
#
#model="vgg19"
#dataset="cifar10"
#init=0
#experiment=1
##solution_string="initial_weights"
#solution_string="test_acc"
#directory=/nobackup/sclaam/checkpoints
#level_1_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_1_${solution_string}.*"))
##level_1_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_1.pth"))
#
#level_2_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_2_${solution_string}.*"))
##level_2_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_2.pth"))
#level_3_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_3_${solution_string}.*"))
##level_3_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_3.pth"))
#level_4_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_4_${solution_string}.*"))
##level_4_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_4.pth"))
#declare -a list_to_use=("${level_1_seeds[@]}")
#
##model="resnet50"
##dataset="tiny_imagenet"
##directory=/nobackup/sclaam/checkpoints
#
##seeds=(0 1 2)
pruning_rates=("0.5" "0.6" "0.7" "0.8")
##rf_levels=(1 4)
##levels_max=${#rf_levels[@]}                                  # Take the length of that array
#seeds_per_level=${#list_to_use[@]}                            # Take the length of that array
number_pruning_rates=${#pruning_rates[@]}                            # Take the length of that array
##for ((idxA=0; idxA<number_pruning_rates; idxA++)); do                # iterate idxA from 0 to length
#for ((idxB=0; idxB<seeds_per_level; idxB++));do              # iterate idxB from 0 to length
#
#qsub -N "${model}_${dataset}pruning_fine_tuning_summary_level_1_${pruning_rates[$idxB]}" run.sh "${model}" "${dataset}" "2" "1" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}" "0.9" "2"
#
##./run.sh "${model}" "${dataset}" "2" "1" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}" "0.5" "1"
#done
##done
#
#declare -a list_to_use=("${level_2_seeds[@]}")
#
#seeds_per_level=${#list_to_use[@]}                            # Take the length of that array
##for ((idxA=0; idxA<levels_max; idxA++)); do                # iterate idxA from 0 to length
##for ((idxA=0; idxA<number_pruning_rates; idxA++)); do                # iterate idxA from 0 to length
#for ((idxB=0; idxB<seeds_per_level; idxB++));do              # iterate idxB from 0 to length
#qsub -N "${model}_${dataset}pruning_fine_tuning_summary_level_2_${pruning_rates[$idxB]}" run.sh "${model}" "${dataset}" "2" "2" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}" "0.9" "2"
##echo "${model}" "${dataset}" "2" "1" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}"
##done
#done
#
#
#declare -a list_to_use=("${level_3_seeds[@]}")
#
#seeds_per_level=${#list_to_use[@]}                            # Take the length of that array
##for ((idxA=0; idxA<levels_max; idxA++)); do                # iterate idxA from 0 to length
##for ((idxA=0; idxA<number_pruning_rates; idxA++)); do                # iterate idxA from 0 to length
#for ((idxB=0; idxB<seeds_per_level; idxB++));do              # iterate idxB from 0 to length
#qsub -N "${model}_${dataset}pruning_fine_tuning_summary_level_3_${pruning_rates[$idxB]}" run.sh "${model}" "${dataset}" "2" "3" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}" "0.9" "2"
##echo "${model}" "${dataset}" "2" "1" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}"
##done
#done
#
#declare -a list_to_use=("${level_4_seeds[@]}")
#
#seeds_per_level=${#list_to_use[@]}                            # Take the length of that array
##for ((idxA=0; idxA<levels_max; idxA++)); do                # iterate idxA from 0 to length
##for ((idxA=0; idxA<number_pruning_rates; idxA++)); do                # iterate idxA from 0 to length
#for ((idxB=0; idxB<seeds_per_level; idxB++));do              # iterate idxB from 0 to length
#qsub -N "${model}_${dataset}pruning_fine_tuning_summary_level_4_${pruning_rates[$idxB]}" run.sh "${model}" "${dataset}" "2" "4" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}" "0.9" "2"
##echo "${model}" "${dataset}" "2" "1" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}"
#done
##done
#
#
#########################################################################################################################
#model="vgg19"
#dataset="tiny_imagenet"
#init=0
##solution_string="initial_weights"
#solution_string="test_acc"
#directory=/nobackup/sclaam/checkpoints
#level_1_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_1_${solution_string}.*"))
##level_1_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_1.pth"))
#
#level_2_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_2_${solution_string}.*"))
##level_2_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_2.pth"))
#level_3_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_3_${solution_string}.*"))
##level_3_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_3.pth"))
#level_4_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_4_${solution_string}.*"))
##level_4_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_4.pth"))
#declare -a list_to_use=("${level_1_seeds[@]}")
#
##model="resnet50"
##dataset="tiny_imagenet"
##directory=/nobackup/sclaam/checkpoints
#
##seeds=(0 1 2)
##rf_levels=(1 4)
##levels_max=${#rf_levels[@]}                                  # Take the length of that array
#seeds_per_level=${#list_to_use[@]}                            # Take the length of that array
##for ((idxA=0; idxA<levels_max; idxA++)); do                # iterate idxA from 0 to length
##for ((idxA=0; idxA<number_pruning_rates; idxA++)); do                # iterate idxA from 0 to length
#for ((idxB=0; idxB<seeds_per_level; idxB++));do              # iterate idxB from 0 to length
#qsub -N "${model}_${dataset}pruning_fine_tuning_summary_level_4_${pruning_rates[$idxB]}" run.sh "${model}" "${dataset}" "2" "1" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}" "0.9" "2"
##echo "${model}" "${dataset}" "2" "1" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}"
##done
#done
#
#
#declare -a list_to_use=("${level_2_seeds[@]}")
#
#seeds_per_level=${#list_to_use[@]}                            # Take the length of that array
##for ((idxA=0; idxA<levels_max; idxA++)); do                # iterate idxA from 0 to length
#
##for ((idxA=0; idxA<number_pruning_rates; idxA++)); do                # iterate idxA from 0 to length
#for ((idxB=0; idxB<seeds_per_level; idxB++));do              # iterate idxB from 0 to length
#qsub -N "${model}_${dataset}pruning_fine_tuning_summary_level_4_${pruning_rates[$idxB]}" run.sh "${model}" "${dataset}" "2" "2" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}" "0.9" "2"
##echo "${model}" "${dataset}" "2" "1" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}"
##done
#done
#
#
#declare -a list_to_use=("${level_3_seeds[@]}")
#
#seeds_per_level=${#list_to_use[@]}                            # Take the length of that array
##for ((idxA=0; idxA<levels_max; idxA++)); do                # iterate idxA from 0 to length
##for ((idxA=0; idxA<number_pruning_rates; idxA++)); do                # iterate idxA from 0 to length
#for ((idxB=0; idxB<seeds_per_level; idxB++));do              # iterate idxB from 0 to length
#qsub -N "${model}_${dataset}pruning_fine_tuning_summary_level_3_${pruning_rates[$idxB]}" run.sh "${model}" "${dataset}" "2" "3" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}" "0.9" "2"
##echo "${model}" "${dataset}" "2" "1" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}"
##done
#done
#
#declare -a list_to_use=("${level_4_seeds[@]}")
#
#seeds_per_level=${#list_to_use[@]}                            # Take the length of that array
##for ((idxA=0; idxA<levels_max; idxA++)); do                # iterate idxA from 0 to length
##for ((idxA=0; idxA<number_pruning_rates; idxA++)); do                # iterate idxA from 0 to length
#for ((idxB=0; idxB<seeds_per_level; idxB++));do              # iterate idxB from 0 to length
#qsub -N "${model}_${dataset}pruning_fine_tuning_summary_level_4_${pruning_rates[$idxB]}" run.sh "${model}" "${dataset}" "2" "4" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}" "0.9" "2"
##echo "${model}" "${dataset}" "2" "1" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}"
##done
#done
#
#
########################################################################################################################


model="vgg19"
dataset="tiny_imagenet"
pruning_rates=("0.5" "0.6" "0.7" "0.8")
number_pruning_rates=${#pruning_rates[@]}                            # Take the length of that array
init=0
#solution_string="initial_weights"
solution_string="test_acc"
directory=/nobackup/sclaam/checkpoints
#level_1_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_1_${solution_string}.*"))
#level_1_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_1.pth"))

level_2_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_2_${solution_string}.*"))
#level_2_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_2.pth"))
level_3_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_3_${solution_string}.*"))
#level_3_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_3.pth"))
level_4_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_4_${solution_string}.*"))
#level_4_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_4.pth"))

#declare -a list_to_use=("${level_1_seeds[@]}")

#model="resnet50"
#dataset="tiny_imagenet"
#directory=/nobackup/sclaam/checkpoints

#seeds=(0 1 2)
##rf_levels=(1 4)
##levels_max=${#rf_levels[@]}                                  # Take the length of that array
#seeds_per_level=${#list_to_use[@]}                            # Take the length of that array
#for ((idxA=0; idxA<number_pruning_rates; idxA++)); do                # iterate idxA from 0 to length
#for ((idxB=0; idxB<seeds_per_level; idxB++));do              # iterate idxB from 0 to length
##qsub -N "${model}_${dataset}pruning_summary_level_1_${pruning_rates[$idxA]}" run.sh "${model}" "${dataset}" "2" "1" "normal" "${directory}" "${pruning_rates[$idxA]}" "1"
###echo "${model}" "${dataset}" "2" "1" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}"
#qsub -N "${model}_${dataset}pruning_summary_level_1" run.sh "${model}" "${dataset}" "2" "1" "normal" "${directory}" "${list_to_use[$idxB]}" "3"
#done
###done
#
#
#declare -a list_to_use=("${level_2_seeds[@]}")
#
#seeds_per_level=${#list_to_use[@]}                            # Take the length of that array
#
##for ((idxA=0; idxA<levels_max; idxA++)); do                # iterate idxA from 0 to length
#
#for ((idxB=0; idxB<seeds_per_level; idxB++));do              # iterate idxB from 0 to length
#
##for ((idxA=0; idxA<number_pruning_rates; idxA++)); do                # iterate idxA from 0 to length
#
#qsub -N "${model}_${dataset}pruning_summary_level_2" run.sh "${model}" "${dataset}" "2" "2" "normal" "${directory}" "${list_to_use[$idxB]}" "3"
#
##echo "${model}" "${dataset}" "2" "1" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}"
##done
#done
#
##
##declare -a list_to_use=("${level_3_seeds[@]}")
##
##seeds_per_level=${#list_to_use[@]}                            # Take the length of that array
###for ((idxA=0; idxA<levels_max; idxA++)); do                # iterate idxA from 0 to length
##for ((idxA=0; idxA<number_pruning_rates; idxA++)); do                # iterate idxA from 0 to length
#for ((idxB=0; idxB<seeds_per_level; idxB++));do              # iterate idxB from 0 to length
##qsub -N "${model}_${dataset}pruning_summary_level_3_${pruning_rates[$idxA]}" run.sh "${model}" "${dataset}" "2" "3" "normal" "${directory}" "${pruning_rates[$idxA]}" "1"
#qsub -N "${model}_${dataset}pruning_summary_level_3" run.sh "${model}" "${dataset}" "2" "3" "normal" "${directory}" "${list_to_use[$idxB]}" "3"
###echo "${model}" "${dataset}" "2" "1" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}"
###done
#done

declare -a list_to_use=("${level_2_seeds[@]}")

seeds_per_level=${#list_to_use[@]}                            # Take the length of that array
#for ((idxA=0; idxA<levels_max; idxA++)); do                # iterate idxA from 0 to length
for ((idxB=0; idxB<seeds_per_level; idxB++));do              # iterate idxB from 0 to length
#for ((idxA=0; idxA<number_pruning_rates; idxA++)); do                # iterate idxA from 0 to length

qsub -N "${model}_${dataset}_fine_Tuning_pruning_level_2_${idxB}" run.sh "${model}" "${dataset}" "2" "2" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}" "0.9" "2"

#qsub -N "${model}_${dataset}pruning_summary_level_4_${idxB}" run.sh "${model}" "${dataset}" "2" "4" "normal" "${directory}" "${list_to_use[$idxB]}" "3"

#echo "${model}" "${dataset}" "2" "1" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}"
done
#done



########################################################################################################################
#
#
#model="resnet50"
#dataset="tiny_imagenet"
#init=0
##solution_string="initial_weights"
#solution_string="test_acc"
#directory=/nobackup/sclaam/checkpoints
##level_1_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_1_${solution_string}.*"))
##level_1_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_1.pth"))
#
#level_2_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_2_${solution_string}.*"))
##level_2_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_2.pth"))
#level_3_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_3_${solution_string}.*"))
##level_3_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_3.pth"))
#level_4_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_4_${solution_string}.*"))
##level_4_seeds=($(ls $directory | grep -i "${model}.*${dataset}.*_level_4.pth"))
##
##declare -a list_to_use=("${level_1_seeds[@]}")
###model="resnet50"
###dataset="tiny_imagenet"
###directory=/nobackup/sclaam/checkpoints
##
###seeds=(0 1 2)
###rf_levels=(1 4)
###levels_max=${#rf_levels[@]}                                  # Take the length of that array
##seeds_per_level=${#list_to_use[@]}                            # Take the length of that array
###for ((idxA=0; idxA<levels_max; idxA++)); do                # iterate idxA from 0 to length
##for ((idxB=0; idxB<seeds_per_level; idxB++));do              # iterate idxB from 0 to length
##qsub -N "${model}_${dataset}pruning_fine_tuning_summary_level_4_${idxB}" run.sh "${model}" "${dataset}" "2" "1" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}"
###echo "${model}" "${dataset}" "2" "1" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}"
##done
#
#
#declare -a list_to_use=("${level_2_seeds[@]}")
#
#seeds_per_level=${#list_to_use[@]}                            # Take the length of that array
##for ((idxA=0; idxA<levels_max; idxA++)); do                # iterate idxA from 0 to length
##for ((idxA=0; idxA<number_pruning_rates; idxA++)); do                # iterate idxA from 0 to length
#for ((idxB=0; idxB<seeds_per_level; idxB++));do              # iterate idxB from 0 to length
#qsub -N "${model}_${dataset}pruning_fine_tuning_summary_level_4_${pruning_rates[$idxB]}" run.sh "${model}" "${dataset}" "2" "2" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}" "0.9" "2"
##echo "${model}" "${dataset}" "2" "1" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}"
##done
#done
#
#
#
#declare -a list_to_use=("${level_3_seeds[@]}")
#
#seeds_per_level=${#list_to_use[@]}                            # Take the length of that array
##for ((idxA=0; idxA<levels_max; idxA++)); do                # iterate idxA from 0 to length
##for ((idxA=0; idxA<number_pruning_rates; idxA++)); do                # iterate idxA from 0 to length
#for ((idxB=0; idxB<seeds_per_level; idxB++));do              # iterate idxB from 0 to length
#qsub -N "${model}_${dataset}pruning_fine_tuning_summary_level_3_${pruning_rates[$idxB]}" run.sh "${model}" "${dataset}" "2" "3" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}" "0.9" "2"
##echo "${model}" "${dataset}" "2" "1" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}"
##done
#done
#
#declare -a list_to_use=("${level_4_seeds[@]}")
#
#seeds_per_level=${#list_to_use[@]}                            # Take the length of that array
##for ((idxA=0; idxA<levels_max; idxA++)); do                # iterate idxA from 0 to length
##for ((idxA=0; idxA<number_pruning_rates; idxA++)); do                # iterate idxA from 0 to length
#for ((idxB=0; idxB<seeds_per_level; idxB++));do              # iterate idxB from 0 to length
#qsub -N "${model}_${dataset}pruning_fine_tuning_summary_level_4_${pruning_rates[$idxB]}" run.sh "${model}" "${dataset}" "2" "4" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}" "0.9" "2"
##echo "${model}" "${dataset}" "2" "1" "normal" "${directory}" "pruning" "${list_to_use[$idxB]}"
##done
#done
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
##done
#
#
###############################################################################
#                 Similarity between seeds loop
###############################################################################

#
#seeds=(0 1)
#rf_levels=(1 2 3 4)
#levels_max=${#rf_levels[@]}                                  # Take the length of that array
#seeds_max=${#seeds[@]}                                  # Take the length of that array
#for ((idxA=0; idxA<levels_max; idxA++)); do              # iterate idxA from 0 to length
#for ((idxB=0; idxB<seeds_max; idxB++)); do              # iterate idxA from 0 to length
#for ((idxC=idxB+1; idxC<seeds_max; idxC++)); do              # iterate idxA from 0 to length
#
###echo "seed_${seeds[$idxB]}_VS_seed_${seeds[$idxC]}_level_${rf_levels[$idxA]}"
##
#qsub -N "similarity_level_${rf_levels[$idxA]}_seeds_${seeds[$idxB]}_${seeds[$idxC]}" run.sh  "vgg19" "_seed_${seeds[$idxB]}_rf_level_${rf_levels[$idxA]}" "_seed_${seeds[$idxC]}_rf_level_${rf_levels[$idxA]}" "alternative" "alternative" "npy" "npy"
##
#done
#done
#done



#######################################################################################################################
#                                    Prune models for a particular RF level, architecture and dataset
#######################################################################################################################



#directory=/nobackup/sclaam/checkpoints
#rf_levels=(2 3 4)
#levels_max=${#rf_levels[@]}                                  # Take the length of that array
#for ((idxA=0; idxA<levels_max; idxA++)); do              # iterate idxA from 0 to length
#
#
#
#
#
#echo "solution ${levels_by_seed[$idxA]}"
#
#
#
#qsub -N "resnet50_tiny_imagenet_gradient_flow_${rf_levels[$idxA]}" run.sh "resnet50" "tiny_imagenet" "2" "${rf_levels[$idxA]}" "normal" "${directory}"
##qsub -N "vgg19_tiny_imagenet_gradient_flow_${rf_levels[$idxA]}" run.sh "vgg19" "tiny_imagenet" "2" "${rf_levels[$idxA]}" "normal" "${directory}"
#
#
#
#
#done


###############################################################################
#                  This is  for changing names
###############################################################################

#directory=/nobackup/sclaam/checkpoints
# all_level_1_seeds=($(ls $directory | grep -i "resnet50_normal_tiny_imagenet.*_level_1_.*" |cut -d_ -f5 |uniq))
# echo $all_level_1_seeds
## all_level_2_seeds=($(ls $directory | grep -i "resnet50_normal_tiny_imagenet.*_level_2_.*" |cut -d_ -f5 |uniq))
## echo $all_level_2_seeds
## all_level_3_seeds=($(ls $directory | grep -i "resnet50_normal_tiny_imagenet.*_level_3_.*" |cut -d_ -f5 |uniq))
## echo $all_level_3_seeds
## all_level_4_seeds=($(ls $directory | grep -i "resnet50_normal_tiny_imagenet.*_level_4_.*" |cut -d_ -f5 |uniq))
## echo $all_level_4_seeds

#declare -a list_to_use=("${all_level_1_seeds[@]}")

#max=${#list_to_use[@]}                                  # Take the length of that array

#echo $max

#for ((idxA=0; idxA<max; idxA++)); do # iterate idxA from 0 to length
#echo "${directory}/.*${list_to_use[$idxA]}\.\*"
#file_names=($(ls $directory | grep -i ".*${list_to_use[$idxA]}.*.pth"))
#echo $file_names
#echo ${#file_names[@]}                                  # Take the length of that array
#echo $idxA
#
#for pathname in  "${file_names[@]}"; do
#replace_string="seed_${idxA}"
#thing="${pathname/"${list_to_use[$idxA]}"/$replace_string}"
#  echo "${thing}"
#  mv -i "${directory}/${pathname}" "${directory}/${thing}"
#done
#done

#
#print_seed_rename () {
#
#max=${#($2[@])}                                  # Take the length of that array
#echo $max
#for ((idxA=0; idxA<max; idxA++)); do # iterate idxA from 0 to length
#echo "$1/.*${$2[$idxA]}\.\*"
#file_names=($(ls $2 | grep -i ".*${$2[$idxA]}.*.pth"))
#echo $file_names
#echo ${#file_names[@]}                                  # Take the length of that array
#echo $idxA
#
#for pathname in  "${file_names[@]}"; do
#replace_string="seed_${idxA}"
#thing="${pathname/"${$2[$idxA]}"/$replace_string}"
#  echo "${thing}"
##  mv -i "${directory}/${pathname}" "${directory}/${thing}"
##
##    if [[ -f $pathname ]] && grep -q -F "$string" "$pathname"; then
##        mv -i "$pathname" "${pathname%.*}.xml"
##    fi
#done
#done
#}

#print_seed_rename $directory $all_level_2_seeds






