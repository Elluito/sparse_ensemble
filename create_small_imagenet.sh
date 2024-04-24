#!/bin/bash -l
small_imagenet_train_folder="/nobackup/sclaam/data/small_imagenet/train"

small_imagenet_val_folder="/nobackup/sclaam/data/small_imagenet/val"

imagenet_train="/nobackup/sclaam/data/imagenet/train"

imagenet_val="/nobackup/sclaam/data/imagenet/val/ILSVRC2012_img_val"

list_of_classes="/nobackup/sclaam/data/wnids.txt"

#cd small_imagenet_foder

while read class; do

    mkdir "${small_imagenet_train_foder}/$class"
#    echo "${small_imagenet_train_folder}/$class"

    # Train images

    train_files=($(ls "${imagenet_train}/$class" | sort -R | tail -500))

    max=${#train_files[@]}                                  # Take the length of that array

    for ((idxA=0; idxA<max; idxA++)); do              # iterate idxA from 0 to length
    # Something involving $file, or you can leave

    # off the while to just get the filenames


    echo "${imagenet_train}/$class/${train_files[$idxA]} ---> ${small_imagenet_train_folder}/$class/"
    idxA=$max+1
#    mv -i "${imagenet_train}/$p/${train_files[$idxA]}" "${small_imagenet_train_folder}/$p/"
#    cp -i "${imagenet_train}/$p/${train_files[$idxA]}" "${small_imagenet_train_folder}/$p/"

    done

    # Test images

    test_files=($(ls "${imagenet_val}/$class" | sort -R | tail -50))

    mkdir "${small_imagenet_val_folder}/$class"
#    echo "${small_imagenet_val_folder}/$class"

    max=${#test_files[@]}                                  # Take the length of that array

    for ((idxA=0; idxA<max; idxA++)); do              # iterate idxA from 0 to length
    # Something involving $file, or you can leave

    # off the while to just get the filenames


    echo "${imagenet_val}/$class/${test_files[$idxA]} ---> ${small_imagenet_val_folder}/$class/"
    idxA=$max+1
#    mv -i "${imagenet_val}/$p/${test_files[$idxA]}" "${small_imagenet_val_folder}/$p/"
#    cp -i "${imagenet_val}/$p/${test_files[$idxA]}" "${small_imagenet_val_folder}/$p/"

    done




done<$list_of_classes