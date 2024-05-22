#!/bin/bash -l



#
#qsub -l h_rt=6:00:00 -l coproc_p100=1 -N "Kfac_optim_rf_1_cifar10_rs" arc3_run.sh "cifar10" "resnet50" "1" "normal" "kfac_optim_hyper" "1"
#qsub -l h_rt=6:00:00 -l coproc_p100=1 -N "Kfac_optim_rf_2_cifar10_rs" arc3_run.sh "cifar10" "resnet50" "2" "normal" "kfac_optim_hyper" "1"
#qsub -l h_rt=6:00:00 -l coproc_p100=1 -N "Kfac_optim_rf_3_cifar10_rs" arc3_run.sh "cifar10" "resnet50" "3" "normal" "kfac_optim_hyper" "1"
qsub -l h_rt=6:00:00 -l coproc_p100=1 -N "Kfac_optim_rf_4_cifar10_rs" arc3_run.sh "cifar10" "resnet50" "4" "normal" "kfac_optim_hyper" "1"
qsub -l h_rt=6:00:00 -l coproc_p100=1 -N "Kfac_optim_rf_k6_cifar10_rs" arc3_run.sh "cifar10" "resnet50" "k6" "normal" "kfac_optim_hyper" "1"
qsub -l h_rt=6:00:00 -l coproc_p100=1 -N "Kfac_optim_rf_k7_cifar10_rs" arc3_run.sh "cifar10" "resnet50" "k7" "normal" "kfac_optim_hyper" "1"
qsub -l h_rt=6:00:00 -l coproc_p100=1 -N "Kfac_optim_rf_k8_cifar10_rs" arc3_run.sh "cifar10" "resnet50" "k8" "normal" "kfac_optim_hyper" "1"
qsub -l h_rt=6:00:00 -l coproc_p100=1 -N "Kfac_optim_rf_k9_cifar10_rs" arc3_run.sh "cifar10" "resnet50" "k9" "normal" "kfac_optim_hyper" "1"



#qsub -l h_rt=30:00:00 -l coproc_p100=1 -N "sam_hyper_parameter_Optim" run.sh "sam"
#qsub -l h_rt=30:00:00 -l coproc_p100=1 -N "kfac_hyper_parameter_Optim" run.sh "kfac"

qsub -l h_rt=6:00:00 -l coproc_p100=1 -N "sam_optim_rf_1_cifar10_rs" arc3_run.sh "cifar10" "resnet50" "1" "normal" "sam_optim_hyper" "2"
qsub -l h_rt=6:00:00 -l coproc_p100=1 -N "sam_optim_rf_2_cifar10_rs" arc3_run.sh "cifar10" "resnet50" "2" "normal" "sam_optim_hyper" "2"
qsub -l h_rt=6:00:00 -l coproc_p100=1 -N "sam_optim_rf_3_cifar10_rs" arc3_run.sh "cifar10" "resnet50" "3" "normal" "sam_optim_hyper" "2"
qsub -l h_rt=6:00:00 -l coproc_p100=1 -N "sam_optim_rf_4_cifar10_rs" arc3_run.sh "cifar10" "resnet50" "4" "normal" "sam_optim_hyper" "2"
qsub -l h_rt=6:00:00 -l coproc_p100=1 -N "sam_optim_rf_k6_cifar10_rs" arc3_run.sh "cifar10" "resnet50" "k6" "normal" "sam_optim_hyper" "2"
qsub -l h_rt=6:00:00 -l coproc_p100=1 -N "sam_optim_rf_k7_cifar10_rs" arc3_run.sh "cifar10" "resnet50" "k7" "normal" "sam_optim_hyper" "2"
qsub -l h_rt=6:00:00 -l coproc_p100=1 -N "sam_optim_rf_k8_cifar10_rs" arc3_run.sh "cifar10" "resnet50" "k8" "normal" "sam_optim_hyper" "2"
qsub -l h_rt=6:00:00 -l coproc_p100=1 -N "sam_optim_rf_k9_cifar10_rs" arc3_run.sh "cifar10" "resnet50" "k9" "normal" "sam_optim_hyper" "2"

