environment = "tslurm"
exclude = "guppy22,dgx1"
for supervised in [False]:  #[False, True]:
    for universal in [True]:  #[False, True]:
        if universal:
            learning_rates = ["1e-4"]
        else:
            learning_rates = ["1e-5"]
        for learning_rate in learning_rates:
            for task in ['graph', 'sum', 'iter', 'eval', 'equals']:
                job_name = (task + ("_universal" if universal else "") +
                            ("_supervised"
                             if supervised else "") + ("_" + learning_rate
                                                       if universal else ""))
                cmd = []
                cmd += ["python ~/code/veclib/scripts/exp.py"]
                cmd += ["--async"]
                if exclude:
                    cmd += ["--exclude " + exclude]
                if supervised:
                    cmd += ["--n_gpus 1"]
                else:
                    cmd += ["--n_gpus 2"]
                cmd += [
                    job_name, "amplification", environment,
                    "amplification/run.py"
                ]
                cmd += ["--task.name " + task]
                cmd += ["--train.num_steps 100000"]
                # cmd += ["--train.warmup_time 20"]
                if universal:
                    cmd += ["--model.joint.universal_transformer t"]
                    cmd += ["--model.joint.learning_rate " + learning_rate]
                if supervised:
                    cmd += ["--train.supervised t"]

                print(" ".join(cmd))
            # print("sleep 10h")