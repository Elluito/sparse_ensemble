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
# For over pruning level
for pr in ${pruning[@]}; do
# For over the models
for model in ${models[@]};do
# For over  datasets
for dataset in ${datasets[@]};do
# For over pruning methods
for pruner in ${pruners[@]};do

for sigma in ${noise[@]}; do
#if [ "${pr}" = "0.8" ] && [ "${sigma}" = "0.01" ]; then
#  continue
#fi
#if [ "${pr}" = "0.9" ] && [  "${model}" = "resnet18" ]; then
# continue
#fi
#echo

git add "gradientflow_stochastic_${pruner}_${model}_${dataset}_sigma_${sigma}_pr${pr}.csv"

done
git add "gradientflow_deterministic_${pruner}_${model}_${dataset}_pr${pr}.csv"
done
done
done
done
