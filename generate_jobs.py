from itertools import product
#xargs -I % zsh -c "%"
environment = "tslurm"
exclude = "guppy9"
with open("jobs.txt", "w") as outfile:
    for (supervised, universal, learning_rate, task) in \
            product([True], [True], ["1e-5"], ['graph', 'sum', 'iter', 'eval', 'equals']):
        if task == "sum":
            learning_rate = "1e-5"
        # ['graph', 'sum', 'iter', 'eval', 'equals']
        # for min_clauses in ["1", "2", "8"]:
        job_name = (
            task + "_big" + ("_universal" if universal else "") +
            ("_supervised"
             if supervised else "") + ("_" + learning_rate
                                       if learning_rate != "1e-5" else "")
            # + ("_" + min_clauses) + "_long")
        )
        cmd = []
        cmd += ["exp.py"]
        # cmd += ["--async"]
        if exclude:
            cmd += ["--exclude " + exclude]
        if supervised:
            cmd += ["--n_gpus 1"]
        else:
            cmd += ["--n_gpus 2"]
        cmd += ["--days 2"]
        cmd += [job_name, "amplification", environment, "amplification/run.py"]
        cmd += ["--task.name " + task]
        # cmd += ["--train.num_steps 400000"]
        # cmd += ["--train.warmup_time 20"]
        if universal:
            cmd += ["--model.joint.universal_transformer t"]
        if supervised:
            cmd += ["--train.supervised t"]
        if learning_rate != "1e-5":
            cmd += ["--model.joint.learning_rate " + learning_rate]
        # cmd += ["--task.min_clauses " + min_clauses]
        # cmd += ["--train.num_steps 10"]
        depth = 12
        answer_depth = 6
        nh = 512
        cmd += [
            """--model.answerer.depth {} --model.answerer.answer_depth {} --model.answerer.nh {}""".
            format(depth, answer_depth, nh)
        ]
        outfile.write(" ".join(cmd) + "\n")
        # print("sleep 10h")

    # depth = 6
    # answer_depth = 3
    # nh = 512
    # for nh in [512, 768, 1024]:
    #     cmd = (
    #         """exp.py --n_gpus 2 gpu_memory_universal amplification tslurm amplification/run.py --task.name eval --train.num_steps 10 --model.joint.universal_transformer t """
    #         +
    #         """--model.answerer.depth {} --model.answerer.answer_depth {} --model.answerer.nh {}""".
    #         format(depth, answer_depth, nh))
    #     print(cmd)
    # depth = 6
    # answer_depth = 3
    # nh = 512
    # for depth in range(6, 12):
    #     cmd = (
    #         """exp.py --n_gpus 2 gpu_memory_universal amplification tslurm amplification/run.py --task.name eval --train.num_steps 10 --model.joint.universal_transformer t """
    #         +
    #         """--model.answerer.depth {} --model.answerer.answer_depth {} --model.answerer.nh {}""".
    #         format(depth, answer_depth, nh))
    #     print(cmd)
    # depth = 6
    # answer_depth = 3
    # nh = 512
    # for answer_depth in range(3, 6):
    #     cmd = (
    #         """exp.py --n_gpus 2 gpu_memory_universal amplification tslurm amplification/run.py --task.name eval --train.num_steps 10 --model.joint.universal_transformer t """
    #         +
    #         """--model.answerer.depth {} --model.answerer.answer_depth {} --model.answerer.nh {}""".
    #         format(depth, answer_depth, nh))
    #     print(cmd)
