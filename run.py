#!/usr/bin/env python3
import argparse
import subprocess
import os
import sys

args = sys.argv
if "--test" in args:
    exit(0)

cmd = ["amplification/run.py"]
skip = 1
for s in args:
    if skip > 0:
        skip -= 1
        continue
    if s == "--project_name":
        skip = 1
    elif s == "--out_dir":
        cmd.append("--train.path")
    elif s == "--hparams":
        skip = 1
    else:
        cmd.append(s)

print(os.getcwd(), cmd)
cmd_str = "export PYTHONPATH=\"${PYTHONPATH}:" + os.getcwd(
) + "\"; python {} ".format(" ".join(cmd))
subprocess.run(cmd_str, shell=True)

# parser = argparse.ArgumentParser()
# parser.add_argument('--test', action="store_true")
# parser.add_argument('--hparams', default="", type=str)
# parser.add_argument('--project_name', type=str, default="")
# parser.add_argument('--out_dir', type=str, default=os.getcwd())
# parser.add_argument('rest', nargs=argparse.REMAINDER)
# args = parser.parse_args()

# if args.test:
#     exit(0)
# print(args)
# cmd = [
#     "amplification/run.py", "task.name", args.project_name, "train.path",
#     args.out_dir
# ] + args.rest
# cmd_str = "python {} ".format(" ".join(cmd))
# print(cmd_str)
# # subprocess.run(cmd_str, shell=True)