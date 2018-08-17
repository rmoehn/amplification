#!/bin/bash
#SBATCH --job-name bandit
#SBATCH --nodes=1   # Number of nodes required
#SBATCH --cpus-per-task=4
#SBATCH --time=01-00:00:00  # Wall Clock time (dd-hh:mm:ss)
#SBATCH --gres=gpu:2
#SBATCH --partition gpu
#SBATCH --overcommit

$HOME/code/veclib/scripts/run.py \
amplification_test \
amplification \
run.py \
--task.name eval \
--train.supervised f \
--train.curriculum t