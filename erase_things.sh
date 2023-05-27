#!/bin/bash -l
#Noise level
# shellcheck disable=SC2054
noise=("0.001" "0.003" "0.005")

#Pruning level
pruning=("0.8" "0.85" "0.9" "0.95")
#Models
models=("resnet18" "resnet50" "VGG19")
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
# For over pruning methods
for pruner in ${pruners[@]};do
if [ "${pr}" = "0.8" ] && [ "${sigma}" = "0.01" ]; then
  continue
fi
if [ "${pr}" = "0.9" ] && [  "${model}" = "resnet18" ]; then
 continue
fi

#echo

if [[ $(find . -type f -name "${model}${dataset}${pruner}-${sigma}-${pr}*" | wc -l) -ne 0 ]]; then

erase "${model}${dataset}${pruner}-${sigma}-${pr}"

else
 echo "no file found"

fi

if [[ $(find . -type f -name "${model}${dataset}${pruner}-deterministic-${pr}*" | wc -l) -ne 0 ]]; then

erase "${model}${dataset}${pruner}-deterministic-${pr}"

else
 echo "no file found"

fi

done
done
done
done
done