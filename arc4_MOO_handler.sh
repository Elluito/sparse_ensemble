#!/bin/bash

#o=$a({resnet18,resnet50,vgg19}" "{cifar10,cifar100}" "{tpe,nsga})
for string in {resnet18,resnet50,vgg19}" "{cifar10,cifar100}" "{tpe,nsga};do
  echo "${string}" "${string#*\ }"
done

for model in "resnet18" "resnet50" "vgg19";do
  for dataset in "cifar10" "cifar100";do


      qsub -l h_rt=48:00:00 -l coproc_v100=1 arc4_MOO_RUN.sh "${model}" "${dataset}" "global" "nsga"

  done
done