#!/bin/bash
EXP_NAME="${1}"
N_GPUS="${2}"
shift 2
REST=$@

nvidia-smi

python $HOME/code/veclib/gpu_test.py $N_GPUS

exec ./run_simple.py \
$EXP_NAME \
amplification \
amplification/run.py $REST

#xargs -L 1 -P num_concurrent_jobs srun < jobs.txt
#xargs -L 1 -P 5 srun --partition gpu --time=00-10:00:00 < jobs.txt
#srun --partition gpu --time=00-10:00:00 --job-name eval --mincpus=4 --gres=gpu:2 exp.slurm eval_universal --task.name eval --model.joint.universal_transformer t --model.joint.learning_rate 1e-4
