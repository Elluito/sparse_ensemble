#!/bin/bash

#--gres=gpu:1

slurm_run.sh "resnet34"
slurm_run.sh "legacy_seresnet34"
#slurm_run.sh "skresnet34"
#slurm_run.sh "mobilenetv2"
slurm_run.sh "mobilenetv3"
slurm_run.sh "efficientnet"
