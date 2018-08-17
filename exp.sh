#!/bin/zsh
srun <<EOF run.py \
amplification_test \
amplification \
run.py \
--task.name eval \
--train.supervised f \
--train.curriculum t
EOF