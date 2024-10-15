#!/bin/bash

run_change_of_name() {
  directory=$1
  model=$2
  dataset=$3
  level=$4
  name=$5

  if [[ $model == *"_"* ]];
  then
  if [[ $dataset == *"_"* ]];
   then

  all_level_6_seeds=($(ls "${directory}" | grep -i "${model}_normal_${dataset}_.*_level_${level}_.*${name}.*" | cut -d_ -f6 | uniq))
else

  all_level_6_seeds=($(ls "${directory}" | grep -i "${model}_normal_${dataset}_.*_level_${level}_.*${name}.*" | cut -d_ -f5 | uniq))
 fi
 else

  if [[ $dataset == *"_"* ]];
   then

  all_level_6_seeds=($(ls "${directory}" | grep -i "${model}_normal_${dataset}_.*_level_${level}_.*${name}.*" | cut -d_ -f5 | uniq))
else
  all_level_6_seeds=($(ls "${directory}" | grep -i "${model}_normal_${dataset}_.*_level_${level}_.*${name}.*" | cut -d_ -f4 | uniq))

 fi
fi


  echo $all_level_6_seeds

  echo " "
  echo "Level ${level}"
  echo " "

  declare -a list_to_use=("${all_level_6_seeds[@]}")
  #
  max=${#list_to_use[@]} # Take the length of that array
  #
  echo $max
  #
  for ((idxA = 0; idxA < max; idxA++)); do # iterate idxA from 0 to length
    echo "${directory}/.*${list_to_use[$idxA]}\.\*"
    file_names=($(ls $directory | grep -i ".*${list_to_use[$idxA]}.*."))
    echo $file_names
    echo ${#file_names[@]} # Take the length of that array
    echo $idxA

    for pathname in "${file_names[@]}"; do
      replace_string="seed.${idxA}"
      thing="${pathname/"${list_to_use[$idxA]}"/$replace_string}"
      echo "${thing}"
#      echo "${directory}/${pathname} ==> ${directory}/${thing}"
#        mv -i "${directory}/${pathname}" "${directory}/${thing}"
    done
  done
}

run_change_of_name $1 $2 $3 $4 $5
