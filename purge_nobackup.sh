#!/bin/bash -l
#Noise level
# shellcheck disable=SC2054
noise=("0.001" "0.003" "0.005" "0.0032" "0.0043" "0.0065" "0.0076" "0.0087" "0.0098" "0.011")
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
#echo "/nobackup/sclaam/gradient_flow_data/${dataset}/stochastic_GLOBAL/${model}/sigma${sigma}/pr${pr}/"
#ls -1t "/nobackup/sclaam/gradient_flow_data/${dataset}/stochastic_GLOBAL/${model}/sigma${sigma}/pr${pr}/" | head -n -10 | xargs -d '\n' rm -rf --
#echo "/nobackup/sclaam/gradient_flow_data/${dataset}/stochastic_LAMP/${model}/sigma${sigma}/pr${pr}/"
#ls -1t "/nobackup/sclaam/gradient_flow_data/${dataset}/stochastic_LAMP/${model}/sigma${sigma}/pr${pr}/"| head -n -10 | xargs -d '\n' rm -rf --

deter_global_name="/nobackup/sclaam/gradient_flow_data/${dataset}/deterministic_GLOBAL/${model}/sigma0.0/pr${pr}/"
echo $deter_global_name
number_directories=$(ls $deter_global_name | wc -l)
number_directories=$(($number_directories + 0))
echo "Number of individuals in folder ${number_directories}"
if [ $number_directories -gt 10 ] && [ -d $deter_global_name ]; then
cd $deter_global_name
ls -1t deter_global_name | head -n -10 | xargs -d '\n' rm -r --
ls
fi
deter_lamp_name="/nobackup/sclaam/gradient_flow_data/${dataset}/deterministic_LAMP/${model}/sigma0.0/pr${pr}/"
echo $deter_lamp_name
number_directories=$(ls $deter_lamp_name | wc -l)
number_directories=$(($number_directories + 0))
echo "Number of individuals in folder ${number_directories}"
if [ $number_directories -gt 10 ] && [ -d $deter_lamp_name ]; then
cd $deter_lamp_name
pwd
ls -1t | head -n -10 | xargs -d '\n' rm -r --
ls #"/nobackup/sclaam/gradient_flow_data/${dataset}/deterministic_LAMP/${model}/sigma0.0/pr${pr}/"
fi

#ls -1t "/nobackup/sclaam/gradient_flow_data/${dataset}/stochastic_GLOBAL/${model}/sigma${sigma}/pr${pr}/"
#| head -n -10 | xargs -d '\n' rm -rf --

done
done
done
done
#done