#!/bin/bash -l
#Noise level
# shellcheck disable=SC2054
noise=("0.001" "0.003" "0.005")
#noise=("0.003")
#Pruning level
pruning=("0.8" "0.85" "0.9" "0.95")
#pruning=("0.9")
#Models
models=("resnet18" "resnet50" "VGG19")
#models=("resnet18")
#Datasets
datasets=("cifar10" "cifar100")
#Pruning methods
pruners=("global" "lamp")



# For over noise level
# shellcheck disable=SC2068
for sigma in ${noise[@]}; do
# For over pruning level
for pr in ${pruning[@]}; do
# For over the models
for model in ${models[@]};do
# For over  datasets
for dataset in ${datasets[@]};do
#for pruner in ${pruners[@]};do
# For over pruning methods
#if [ "${pr}" = "0.9" ] && [  "${model}" = "resnet18" ]; then
# continue
#fi
#echo

ls -1t "/nobackup/sclaam/gradient_flow_data/${dataset}/stochastic_GLOBAL/${model}/sigma${sigma}/pr${pr}/"
#| head -n -10 | xargs -d '\n' rm -rf --

done
done
done
done
#done