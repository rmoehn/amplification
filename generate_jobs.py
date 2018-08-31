#xargs -I % zsh -c "%"
environment = "tslurm"
exclude = "dgx1,guppy24,guppy20"
for supervised in [False]:  #[False, True]:
    for universal in [False]:
        # if universal:
        #     learning_rates = ["1e-4"]
        # else:
        #     learning_rates = ["1e-5"]
        learning_rates = ["1e-4", "1e-5"]
        for learning_rate in learning_rates:
            for task in ['graph', 'sum', 'iter', 'eval', 'equals']:
                job_name = (task + ("_universal" if universal else "") +
                            ("_supervised" if supervised else "") +
                            ("_" + learning_rate
                             if learning_rate != "1e-5" else ""))
                cmd = []
                cmd += ["exp.py"]
                # cmd += ["--async"]
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
                # cmd += ["--train.num_steps 400000"]
                # cmd += ["--train.warmup_time 20"]
                if universal:
                    cmd += ["--model.joint.universal_transformer t"]
                if supervised:
                    cmd += ["--train.supervised t"]
                if learning_rate != "1e-5":
                    cmd += ["--model.joint.learning_rate " + learning_rate]

                print(" ".join(cmd))
            # print("sleep 10h")