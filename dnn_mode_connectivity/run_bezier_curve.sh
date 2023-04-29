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
python train.py --dir=curves/cifar10/resnet18/global/fine_tuned --dataset=CIFAR10 --use_test --transform=ResNet --data_path=/nobackup/sclaam/data/CIFAR10 --model=ResNet18 --curve=Bezier --num_bends=3 --init_start=/nobackup/sclaam/gradient_flow_data/cifar10/stochastic_GLOBAL/resnet18/sigma0.0021/pr0.9/1677548381.96440/weigths/epoch_90.pth --init_end=/nobackup/sclaam/gradient_flow_data/cifar10/deterministic_GLOBAL/resnet18/0.9/1676918518.39/weigths/ --epochs=200 --lr=0.03 --wd=5e-4 --fix_end --fix_start --num_workers=0
python eval_curve.py --dir=evaluate_curve/cifar10/resnet18/global/fine_tuned --dataset=CIFAR10 --data_path=datasets --transform=ResNet --model=ResNet18 --wd=5e-4 --curve=Bezier --num_bends=3 --ckpt=curves/cifar10/resnet18/global/fine_tuned/checkpoint-200.pt --num_points=60 --use_test --num-workers=0