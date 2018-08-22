for supervised in [False, True]:
    for universal in [True, False]:
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
                cmd = ["sbatch --partition gpu --time=00-10:00:00"]
                cmd += ["-x guppy4,guppy5,guppy6,guppy7,guppy10"]
                if supervised:
                    cmd += ["--gres=gpu:1 --cpus-per-task=4"]
                else:
                    cmd += ["--gres=gpu:2 --cpus-per-task=8"]
                cmd += ["--job-name " + job_name]
                cmd += ["exp.slurm"]
                cmd += [job_name]
                if supervised:
                    cmd += ["1"]
                else:
                    cmd += ["2"]
                # cmd = [
                #     "python ~/code/veclib/exp.py", job_name, "amplification",
                #     "bluebird"
                # ]
                # cmd += ["amplification/run.py"]
                cmd += ["--task.name " + task]
                cmd += ["--train.num_steps 100000"]
                if universal:
                    cmd += ["--model.joint.universal_transformer t"]
                    cmd += ["--model.joint.learning_rate " + learning_rate]
                if supervised:
                    cmd += ["--train.supervised t"]

                print(" ".join(cmd))
            # print("sleep 10h")