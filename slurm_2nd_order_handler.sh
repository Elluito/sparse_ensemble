#!/bin/bash



sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="sam_optim_rf_1_cifar10_rs.err" --output="sam_optim_rf_1_cifar10_rs.output" --job-name="sam_optim_rf_1_cifar10_rs" slurm_2nd_order_run.sh "cifar10" "resnet50" "1" "normal" "sam_optim_hyper_200" "2"
sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="sam_optim_rf_2_cifar10_rs.err" --output="sam_optim_rf_2_cifar10_rs.output" --job-name="sam_optim_rf_2_cifar10_rs" slurm_2nd_order_run.sh "cifar10" "resnet50" "2" "normal" "sam_optim_hyper_200" "2"
sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="sam_optim_rf_3_cifar10_rs.err" --output="sam_optim_rf_3_cifar10_rs.output" --job-name="sam_optim_rf_3_cifar10_rs" slurm_2nd_order_run.sh "cifar10" "resnet50" "3" "normal" "sam_optim_hyper_200" "2"
sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="sam_optim_rf_4_cifar10_rs.err" --output="sam_optim_rf_4_cifar10_rs.output" --job-name="sam_optim_rf_4_cifar10_rs" slurm_2nd_order_run.sh "cifar10" "resnet50" "4" "normal" "sam_optim_hyper_200" "2"
sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="sam_optim_rf_5_cifar10_rs.err" --output="sam_optim_rf_5_cifar10_rs.output" --job-name="sam_optim_rf_5_cifar10_rs" slurm_2nd_order_run.sh "cifar10" "resnet50" "5" "normal" "sam_optim_hyper_200" "2"
sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="sam_optim_rf_6_cifar10_rs.err" --output="sam_optim_rf_6_cifar10_rs.output" --job-name="sam_optim_rf_6_cifar10_rs" slurm_2nd_order_run.sh "cifar10" "resnet50" "6" "normal" "sam_optim_hyper_200" "2"
sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="sam_optim_rf_7_cifar10_rs.err" --output="sam_optim_rf_7_cifar10_rs.output" --job-name="sam_optim_rf_7_cifar10_rs" slurm_2nd_order_run.sh "cifar10" "resnet50" "7" "normal" "sam_optim_hyper_200" "2"
sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="sam_optim_rf_8_cifar10_rs.err" --output="sam_optim_rf_8_cifar10_rs.output"  --job-name="sam_optim_rf_8_cifar10_rs" slurm_2nd_order_run.sh "cifar10" "resnet50" "8" "normal" "sam_optim_hyper_200" "2"