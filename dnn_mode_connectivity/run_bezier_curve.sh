#!/bin/bash -l
# Submission script for serial Python job
# Luis Alfredo Avendano  2021-10-22

# Run from the current directory and with current environment
#$ -cwd -V

# Ask for some time (hh:mm:ss max of 48:00:00)
#$ -l h_rt=012:00:00

# ASk for some GPU
#$ -l coproc_p100=1

# Ask for some memory (by default, 1G, without a request)
#$ -l h_vmem=12G
# -pe smp 3
# Send emails when job starts and ends
#$ -m e
#export OMP_NUM_THREADS=3
#export MKL_NUM_THREADS=3
#export OPENBLAS_NUM_THREADS=1
# Now run the job
#module load intel openmpi
module add anaconda
module add cuda/11.1.1
conda activate work
which python
#nvcc --version
#~python $1 $2 $3 $4 $5 $6

# For  comparing fine tuned resnet18 on cifar10 with global pruning
#python train.py --dir=curves/cifar10/resnet18/global/fine_tuned --dataset=CIFAR10 --use_test --transform=ResNet --data_path=/nobackup/sclaam/data/CIFAR10 --model=ResNet18 --curve=Bezier --num_bends=3 --init_start=/nobackup/sclaam/gradient_flow_data/cifar10/stochastic_GLOBAL/resnet18/0.9/1676842423.42/weigths/epoch_90.pth --init_end=/nobackup/sclaam/gradient_flow_data/cifar10/deterministic_GLOBAL/resnet18/0.9/1676918518.39/weigths/epoch_90.pth --epochs=200 --lr=0.03 --wd=5e-4 --fix_end --fix_start --num_workers=0
#python eval_curve.py --dir=evaluate_curve/cifar10/resnet18/global/fine_tuned --dataset=CIFAR10 --data_path=datasets --transform=ResNet --model=ResNet18 --wd=5e-4 --curve=Bezier --num_bends=3 --ckpt=curves/cifar10/resnet18/global/fine_tuned/checkpoint-200.pt --num_points=60 --use_test --num_workers=0

# This is for the plane on cifar10 on resnet for global


# For  comparing fine tuned resnet18 on cifar10 with lamp pruning
#python train.py --dir=curves/cifar10/resnet18/lamp/fine_tuned --dataset=CIFAR10 --use_test --transform=ResNet --data_path=/nobackup/sclaam/data/CIFAR10 --model=ResNet18 --curve=Bezier --num_bends=3 --init_start=/nobackup/sclaam/gradient_flow_data/cifar10/stochastic_LAMP/resnet18/0.9/1676842555.46/weigths/epoch_90.pth --init_end=/nobackup/sclaam/gradient_flow_data/cifar10/deterministic_LAMP/resnet18/sigma0.0/pr0.9/1677677439.38/weigths/epoch_90.pth --epochs=200 --lr=0.03 --wd=5e-4 --fix_end --fix_start --num_workers=0
#python eval_curve.py --dir=evaluate_curve/cifar10/resnet18/lamp/fine_tuned --dataset=CIFAR10 --data_path=/nobackup/sclaam/data/CIFAR10  --transform=ResNet --model=ResNet18 --wd=5e-4 --curve=Bezier --num_bends=3 --ckpt=curves/cifar10/resnet18/lamp/fine_tuned/checkpoint-200.pt --num_points=60 --use_test --num_workers=0
# This is for the plane on cifar10 on resnet for lamp



# Evaluating plane of resnet18 global cifar10
#python plane.py --dir=evaluate_plane/cifar10/resnet18/global/fine_tuned --dataset=CIFAR10 --data_path=/nobackup/sclaam/data/CIFAR10 --transform=ResNet --model=ResNet18 --wd=5e-4 --curve=Bezier --num_bends=3 --ckpt=curves/cifar10/resnet18/global/checkpoint-200.pt --curve_points=60 --use_test

# VGG19 on CIFAR10

#/nobackup/sclaam/gradient_flow_data/cifar10/stochastic_GLOBAL/VGG19/sigma0.001/pr0.94/1682352848.15739/weigths/epoch_90.pth

#/nobackup/sclaam/gradient_flow_data/cifar10/stochastic_GLOBAL/VGG19/sigma0.005/pr0.94/1682367758.87520/weigths/epoch_90.pth

#/nobackup/sclaam/gradient_flow_data/cifar10/deterministic_GLOBAL/VGG19/sigma0.0/pr0.94/1682283729.46/weigths/epoch_90.pth

# For  comparing fine tuned VGG on cifar10 with global pruning
#python train.py --dir=/nobackup/sclaam/curves/cifar10/vgg19/global/fine_tuned/s0.005 --dataset=CIFAR10 --use_test --transform=VGG --data_path=/nobackup/sclaam/data/CIFAR10 --model=VGG19 --curve=Bezier --num_bends=3 --init_start=/nobackup/sclaam/gradient_flow_data/cifar10/stochastic_GLOBAL/VGG19/sigma0.005/pr0.94/1682367758.87520/weigths/epoch_90.pth --init_end=/nobackup/sclaam/gradient_flow_data/cifar10/deterministic_GLOBAL/VGG19/sigma0.0/pr0.94/1682283729.46/weigths/epoch_90.pth --epochs=200 --lr=0.03 --wd=5e-4 --fix_end --fix_start --num_workers=0
#python eval_curve.py --dir=evaluate_curve/cifar10/vgg19/global/fine_tuned/s0.005 --dataset=CIFAR10 --data_path=/nobackup/sclaam/data/CIFAR10 --transform=VGG --model=VGG19 --wd=5e-4 --curve=Bezier --num_bends=3 --ckpt=/nobackup/sclaam/curves/cifar10/vgg19/global/fine_tuned/s0.005/checkpoint-200.pt --num_points=60 --use_test --num_workers=0

# For  comparing fine tuned VGG on cifar10 with lamp pruning
# sigma0.005 /nobackup/sclaam/gradient_flow_data/cifar10/stochastic_LAMP/VGG19/sigma0.005/pr0.94/1682458386.92052/weigths/epoch_90.pth
# sigma0.001 /nobackup/sclaam/gradient_flow_data/cifar10/stochastic_LAMP/VGG19/sigma0.001/pr0.94/1682357810.56177/weigths/epoch_90.pth

# python train.py --dir=/nobackup/sclaam/curves/cifar10/vgg19/lamp/fine_tuned/s0.005 --dataset=CIFAR10 --use_test --transform=VGG --data_path=/nobackup/sclaam/data/CIFAR10 --model=VGG19 --curve=Bezier --num_bends=3 --init_start=/nobackup/sclaam/gradient_flow_data/cifar10/stochastic_LAMP/VGG19/sigma0.005/pr0.94/1682458386.92052/weigths/epoch_90.pth --init_end=/nobackup/sclaam/gradient_flow_data/cifar10/deterministic_LAMP/VGG19/sigma0.0/pr0.94/1682284485.05/weigths/epoch_90.pth --epochs=200 --lr=0.03 --wd=5e-4 --fix_end --fix_start --num_workers=0
# python eval_curve.py --dir=evaluate_curve/cifar10/vgg19/lamp/fine_tuned/s0.005 --dataset=CIFAR10 --data_path=/nobackup/sclaam/data/CIFAR10 --transform=VGG --model=VGG19 --wd=5e-4 --curve=Bezier --num_bends=3 --ckpt=/nobackup/sclaam/curves/cifar10/vgg19/lamp/fine_tuned/s0.005/checkpoint-200.pt --num_points=60 --use_test --num_workers=0


# RESNET18 ON CIFAR100
# LAMP
# Sigma 0.001 /nobackup/sclaam/gradient_flow_data/cifar100/stochastic_LAMP/resnet18/sigma0.001/pr0.9/1681334666.23798/weigths/epoch_100.pth
# deterministic /nobackup/sclaam/gradient_flow_data/cifar100/deterministic_LAMP/resnet18/sigma0.0/pr0.9/1681330063.39/weigths/epoch_100.pth
#python train.py --dir=curves/cifar100/resnet18/lamp/fine_tuned --dataset=CIFAR100 --use_test --transform=ResNet --data_path=/nobackup/sclaam/data/CIFAR100 --model=ResNet18 --curve=Bezier --num_bends=3 --init_start=/nobackup/sclaam/gradient_flow_data/cifar100/stochastic_LAMP/resnet18/sigma0.001/pr0.9/1681334666.23798/weigths/epoch_100.pth --init_end=/nobackup/sclaam/gradient_flow_data/cifar100/deterministic_LAMP/resnet18/sigma0.0/pr0.9/1681330063.39/weigths/epoch_100.pth --epochs=200 --lr=0.03 --wd=5e-4 --fix_end --fix_start --num_workers=0
#python eval_curve.py --dir=evaluate_curve/cifar100/resnet18/lamp/fine_tuned --dataset=CIFAR100 --data_path=/nobackup/sclaam/data/CIFAR100 --transform=ResNet --model=ResNet18 --wd=5e-4 --curve=Bezier --num_bends=3 --ckpt=curves/cifar100/resnet18/lamp/fine_tuned/checkpoint-200.pt --num_points=60 --use_test --num_workers=0

# GLOBAL
# Sigma 0.001 /nobackup/sclaam/gradient_flow_data/cifar100/stochastic_GLOBAL/resnet18/sigma0.001/pr0.9/1681333033.73189/weigths/epoch_100.pth
# deterministic /nobackup/sclaam/gradient_flow_data/cifar100/deterministic_GLOBAL/resnet18/sigma0.0/pr0.9/1681329773.24/weigths/epoch_100.pth

python train.py --dir=curves/cifar100/resnet18/global/fine_tuned --dataset=CIFAR100 --use_test --transform=ResNet --data_path=/nobackup/sclaam/data/CIFAR100 --model=ResNet18 --curve=Bezier --num_bends=3 --init_start=/nobackup/sclaam/gradient_flow_data/cifar100/stochastic_GLOBAL/resnet18/sigma0.001/pr0.9/1681333033.73189/weigths/epoch_100.pth --init_end=/nobackup/sclaam/gradient_flow_data/cifar100/deterministic_GLOBAL/resnet18/sigma0.0/pr0.9/1681329773.24/weigths/epoch_100.pth --epochs=200 --lr=0.03 --wd=5e-4 --fix_end --fix_start --num_workers=0
python eval_curve.py --dir=evaluate_curve/cifar100/resnet18/global/fine_tuned --dataset=CIFAR100 --data_path=/nobackup/sclaam/data/CIFAR100 --transform=ResNet --model=ResNet18 --wd=5e-4 --curve=Bezier --num_bends=3 --ckpt=curves/cifar100/resnet18/global/fine_tuned/checkpoint-200.pt --num_points=60 --use_test --num_workers=0



