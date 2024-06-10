#!/bin/bash -l
directory=$HOME/checkpoints
search_string="resnet_small_normal_small_imagenet.*_level_4_.*recording_200.*"
all_level_3_seeds=($(ls $directory | grep -i "" |cut -d_ -f5 |uniq))
echo $all_level_3_seeds



echo " "
echo "Level 5 \n"
echo " "

declare -a list_to_use=("${all_level_3_seeds[@]}")

max=${#list_to_use[@]}                                  # Take the length of that array

echo $max

for ((idxA=0; idxA<max; idxA++)); do # iterate idxA from 0 to length
echo "${directory}/.*${list_to_use[$idxA]}\.\*"
file_names=($(ls $directory | grep -i ".*${list_to_use[$idxA]}.*.pth"))
echo $file_names
echo ${#file_names[@]}                                  # Take the length of that array
echo $idxA

for pathname in  "${file_names[@]}"; do
replace_string="seed.${idxA}"
thing="${pathname/"${list_to_use[$idxA]}"/$replace_string}"
  echo "${thing}"
  echo "${directory}/${pathname} ===> ${directory}/${thing}"
#  mv -i "${directory}/${pathname}" "${directory}/${thing}"

done
done
