#!/usr/bin/env python3
import argparse
import subprocess
import os
from pathlib import Path
import flock
import pytimeparse

parser = argparse.ArgumentParser()
parser.add_argument('name', type=str)
parser.add_argument('code_dir_relative', type=str)
parser.add_argument('--min_free_space', type=int, default=1e9)
parser.add_argument('rest', nargs=argparse.REMAINDER)
args = parser.parse_args()

code_home = os.path.join(str(Path.home()), "code")
os.chdir(code_home)
code_dir = os.path.join(code_home, args.code_dir_relative)

experiment_home = None
experiment_home_candidates = ["/scratch/gobi1/william/"]
for c in experiment_home_candidates:
    if os.path.exists(c):
        experiment_home = c
        break
if experiment_home is None:
    print("Could not find experiment home directory")
    exit(1)

experiment_dir_base = os.path.join(experiment_home, args.code_dir_relative,
                                   args.name)
os.makedirs(experiment_dir_base, exist_ok=True)

statvfs = os.statvfs(experiment_dir_base)
freespace = statvfs.f_frsize * statvfs.f_bavail
if freespace < args.min_free_space:
    print("Not enough free space: ", freespace, args.min_free_space)
    exit(1)

with open(os.path.join(experiment_dir_base, ".lockfile"), 'w') as fp:
    with flock.Flock(fp, flock.LOCK_EX) as lock:
        i = 0
        while True:
            experiment_dir = os.path.join(experiment_dir_base, str(i))
            if not os.path.exists(experiment_dir):
                break
            i += 1
        print(experiment_dir)
        os.makedirs(experiment_dir)
        link_name = os.path.join(experiment_dir_base, "latest")
        if os.path.exists(link_name):
            os.unlink(link_name)
        os.symlink(str(i), link_name, True)

cmd = """rsync -avz """
rsync_exclude_file = os.path.join(code_dir, "rsync_exclude.txt")
if os.path.exists(rsync_exclude_file):
    cmd += """--exclude-from={rsync_exclude_file} """.format(
        rsync_exclude_file=rsync_exclude_file)
cmd += """ {code_dir}/ {experiment_dir}""".format(
    code_dir=code_dir, experiment_dir=experiment_dir)
subprocess.run(cmd, shell=True)
print(args.name)
print(args.rest)


def run(experiment_dir, run_name, cmd):
    # cmd[0] = os.path.join(experiment_dir, cmd[0])
    out_dir = os.path.join(experiment_dir, run_name)
    os.makedirs(out_dir)
    cmd = cmd + ["--train.path", out_dir]
    cmd_str = ("export PYTHONPATH=\"${PYTHONPATH}:\"" +
               os.getcwd() + " ; python {} | tee {}".format(
                   " ".join(cmd), os.path.join(experiment_dir, "log.txt")))
    print("cd " + experiment_dir)
    print(cmd_str)
    try:
        subprocess.run(cmd_str, shell=True, cwd=experiment_dir, check=True)
    except subprocess.CalledProcessError:
        print("Failed command: {}".format(cmd_str))
        raise

    return None


run(
    experiment_dir,
    "0",
    args.rest,
)
