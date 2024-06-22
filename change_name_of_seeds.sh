#!/bin/bash

run_change_of_name() {
  directory=$1
  model=$2
  dataset=$3
  level=$4
  name=$5
  all_level_6_seeds=($(ls "${directory}" | grep -i "${model}_normal_${dataset}_.*_level_${level}_.*${name}.*" | cut -d_ -f4 | uniq))
  echo $all_level_6_seeds

  echo " "
  echo "Level ${level} \n"
  echo " "

  declare -a list_to_use=("${all_level_7_seeds[@]}")
  #
  max=${#list_to_use[@]} # Take the length of that array
  #
  echo $max
  #
  for ((idxA = 0; idxA < max; idxA++)); do # iterate idxA from 0 to length
    echo "${directory}/.*${list_to_use[$idxA]}\.\*"
    file_names=($(ls $directory | grep -i ".*${list_to_use[$idxA]}.*.pth"))
    echo $file_names
    echo ${#file_names[@]} # Take the length of that array
    echo $idxA

    for pathname in "${file_names[@]}"; do
      replace_string="seed_${idxA}"
      thing="${pathname/"${list_to_use[$idxA]}"/$replace_string}"
      echo "${thing}"
      #  mv -i "${directory}/${pathname}" "${directory}/${thing}"
    done
  done
}

run_change_of_name $1 "resnet_small" "small_imagenet" $2 $3
