#!/bin/bash

sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="kfac_optim_rf_1_cifar10_rs.err" --output="kfac_optim_rf_1_cifar10_rs.output"  --job-name="Kfac_optim_rf_1_cifar10_rs" slurm_run.sh "cifar10" "resnet50" "1" "normal" "kfac_optim_hyper" "1"
sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="kfac_optim_rf_2_cifar10_rs.err" --output="kfac_optim_rf_2_cifar10_rs.output" --job-name="Kfac_optim_rf_2_cifar10_rs" slurm_run.sh "cifar10" "resnet50" "2" "normal" "kfac_optim_hyper" "1"
sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="kfac_optim_rf_3_cifar10_rs.err" --output="kfac_optim_rf_3_cifar10_rs.output"  --job-name="Kfac_optim_rf_3_cifar10_rs" slurm_run.sh "cifar10" "resnet50" "3" "normal" "kfac_optim_hyper" "1"
sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="kfac_optim_rf_4_cifar10_rs.err" --output="kfac_optim_rf_4_cifar10_rs.output" --job-name="Kfac_optim_rf_4_cifar10_rs" slurm_run.sh "cifar10" "resnet50" "4" "normal" "kfac_optim_hyper" "1"
sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="kfac_optim_rf_k6_cifar10_rs.err" --output="kfac_optim_rf_k6_cifar10_rs.output" --job-name="Kfac_optim_rf_k6_cifar10_rs" slurm_run.sh "cifar10" "resnet50" "k6" "normal" "kfac_optim_hyper" "1"
sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="kfac_optim_rf_k7_cifar10_rs.err" --output="kfac_optim_rf_k7_cifar10_rs.output" --job-name="Kfac_optim_rf_k7_cifar10_rs" slurm_run.sh "cifar10" "resnet50" "k7" "normal" "kfac_optim_hyper" "1"
sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="kfac_optim_rf_k8_cifar10_rs.err" --output="kfac_optim_rf_k8_cifar10_rs.output" --job-name="Kfac_optim_rf_k8_cifar10_rs" slurm_run.sh "cifar10" "resnet50" "k8" "normal" "kfac_optim_hyper" "1"
sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="kfac_optim_rf_k9_cifar10_rs.err" --output="kfac_optim_rf_k9_cifar10_rs.output" --job-name="Kfac_optim_rf_k9_cifar10_rs" slurm_run.sh "cifar10" "resnet50" "k9" "normal" "kfac_optim_hyper" "1"




sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="sam_optim_rf_1_cifar10_rs.err" --output="sam_optim_rf_1_cifar10_rs.output" --job-name="sam_optim_rf_1_cifar10_rs" slurm_run.sh "cifar10" "resnet50" "1" "normal" "sam_optim_hyper" "2"
sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="sam_optim_rf_2_cifar10_rs.err" --output="sam_optim_rf_2_cifar10_rs.output" --job-name="sam_optim_rf_2_cifar10_rs" slurm_run.sh "cifar10" "resnet50" "2" "normal" "sam_optim_hyper" "2"
sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="sam_optim_rf_3_cifar10_rs.err" --output="sam_optim_rf_3_cifar10_rs.output" --job-name="sam_optim_rf_3_cifar10_rs" slurm_run.sh "cifar10" "resnet50" "3" "normal" "sam_optim_hyper" "2"
sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="sam_optim_rf_4_cifar10_rs.err" --output="sam_optim_rf_4_cifar10_rs.output" --job-name="sam_optim_rf_4_cifar10_rs" slurm_run.sh "cifar10" "resnet50" "4" "normal" "sam_optim_hyper" "2"
sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="sam_optim_rf_k6_cifar10_rs.err" --output="sam_optim_rf_k6_cifar10_rs.output" --job-name="sam_optim_rf_k6_cifar10_rs" slurm_run.sh "cifar10" "resnet50" "k6" "normal" "sam_optim_hyper" "2"
sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="sam_optim_rf_k7_cifar10_rs.err" --output="sam_optim_rf_k7_cifar10_rs.output" --job-name="sam_optim_rf_k7_cifar10_rs" slurm_run.sh "cifar10" "resnet50" "k7" "normal" "sam_optim_hyper" "2"
sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="sam_optim_rf_k8_cifar10_rs.err" --output="sam_optim_rf_k8_cifar10_rs.output" --job-name="sam_optim_rf_k8_cifar10_rs" slurm_run.sh "cifar10" "resnet50" "k8" "normal" "sam_optim_hyper" "2"
sbatch --nodes=1 --time=24:00:00 --partition=small --gres=gpu:1 --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="sam_optim_rf_k9_cifar10_rs.err" --output="sam_optim_rf_k9_cifar10_rs.output"  --job-name="sam_optim_rf_k9_cifar10_rs" slurm_run.sh "cifar10" "resnet50" "k9" "normal" "sam_optim_hyper" "2"

#sbatch --nodes=1 --time=00:05:00 --partition=small  --mail-type=ALL --mail-user=sclaam@leeds.ac.uk --error="import_test.err" --gres=gpu:1 --output="import_test.output"  --job-name="import_test" slurm_run.sh

#--gres=gpu:1
