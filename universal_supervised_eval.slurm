#!/bin/bash
#SBATCH --job-name bandit
#SBATCH --nodes=1   # Number of nodes required
#SBATCH --cpus-per-task=2
#SBATCH --time=01-00:00:00  # Wall Clock time (dd-hh:mm:ss)
#SBATCH --gres=gpu:1
#SBATCH --partition gpu
#SBATCH --overcommit

$HOME/code/veclib/scripts/run.py \
universal_supervised_eval \
amplification \
run.py \
--task.name eval \
--train.supervised t \
--train.curriculum t \
--model.joint.universal_transformer t