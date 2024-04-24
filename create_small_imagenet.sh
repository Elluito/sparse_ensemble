#!/bin/bash -l
small_imagenet_train_folder="nobackup/sclaam/data/small_imagenet/train"

small_imagenet_val_folder="nobackup/sclaam/data/small_imagenet/val"

imagenet_train="nobackup/sclaam/data/imagenet/train"

imagenet_val="nobackup/sclaam/data/imagenet/val"

list_of_classes="nobackup/sclaam/data/windt.txt"

#cd small_imagenet_foder

while read p; do

    mkdir -p "${small_imagenet_train_foder}/$p"

    # Train images

    train_files=($(ls "${imagenet_train}/$p" | sort -R | tail -500))

    max=${#train_files[@]}                                  # Take the length of that array

    for ((idxA=0; idxA<max; idxA++)); do              # iterate idxA from 0 to length
    # Something involving $file, or you can leave

    # off the while to just get the filenames


    echo "${imagenet_train}/$p/${train_files[$idxA]}--->${small_imagenet_train_folder}/$p/"
#    mv -i "${imagenet_train}/$p/${train_files[$idxA]}" "${small_imagenet_train_folder}/$p/"

    done

    # Test images

    test_files=($(ls "${imagenet_val}/$p" | sort -R | tail -50))

    max=${#test_files[@]}                                  # Take the length of that array

    for ((idxA=0; idxA<max; idxA++)); do              # iterate idxA from 0 to length
    # Something involving $file, or you can leave

    # off the while to just get the filenames


    echo "${imagenet_val}/$p/${test_files[$idxA]} ---> ${small_imagenet_val_folder}/$p/"
#    mv -i "${imagenet_val}/$p/${test_files[$idxA]}" "${small_imagenet_val_folder}/$p/"

    done




done<$list_of_classes