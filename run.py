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