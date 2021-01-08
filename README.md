Content of original README:

> Do not expect this code to run.
> It was written by Paul Christiano and Buck Shlegeris to produce the results in the paper
> `Supervising Strong Learners by Amplifying Weak Experts`.
> It is not intended to allow other researchers to reproduce those results,
> and won't be maintained or improved.
> It is released under the MIT license.


William Saunders (@william-r-s) and I have adapted the code and made it
runnable. For now I have little time to make it runnable easily (by documenting
it in detail), but I'll give you some hints. Feel free to email me or open
issues if you need more help.

See also: [IDA with RL and overseer failures](https://github.com/rmoehn/farlamp)


How to run the code
-------------------

1. Create a conda environment and activate it.
2. Install Tensorflow 2.0 and Numpy. I also recommend IPython. If you later get
   weird OMP errors, you need to install `nomkl`. (See also [‘OMP: Error #15:
   Initializing libiomp5.dylib, but found libiomp5.dylib already initialized’ on
   StackOverflow](https://stackoverflow.com/questions/53648730/omp-error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-in).)
3. If TensorBoard doesn't work properly within the conda environment, I
   recommend installing it into a virtualenv somewhere else.
4. Run the following:

    ```shell
    python experiment.py --mode local -e iterate_rm -n iterate
    ```

   This runs a tiny model for a few steps on the permutation powering task. If
   you have a GPU or fast CPU, this should finish in about a minute. Of course,
   the accuracy will still be low. For close to 100 % accuracy you need about
   4000 steps.

   If you replace `iterate_rm` with `iterate_v1` or `iterate_fail_v1`, you'll
   hopefully reproduce the results I got in my report
   [Training a tiny SupAmp model on easy tasks: The influence of failure rate on
   learning curves](https://github.com/rmoehn/farlamp/blob/master/tiny-supfail.pdf).

When you run an experiment, it saves TensorBoard-compatible logs in the
directory `results`. It saves a file with hyperparameters along with the logs.
And it tags the commit at which the experiments were run. If there were
uncommitted changes, it commits them first.


View results in TensorBoard
---------------------------

1. The TensorBoard Custom Scalars plugin (no separate installation needed)
   allows you to see several metrics in one plot. To write a layout definition
   to where the plugin can pick it up, run this Python code:

    ```python
    from amplification import tb_layout
    tb_layout.write(tb_layout.main_layout, "results/iterate")
    ```

2. Start TensorBoard (assuming you've installed it into a virtualenv called
   `tbvenv`):

    ```shell
    ./tbvenv/bin/tensorboard --log-dir results
    ```


Rough changelog, compared to the original code
----------------------------------------------

- William added a SAT task and the option to use a Universal Transformer instead
  of a Transformer. Plus some other things. Just look at the first commits in
  the Git history and you'll know as much as I know.
- Add clarifying comments, especially in `amplification.tasks.iterate`.
- Make it runnable without a GPU. If you run the tiny model on tiny tasks, it
  should run on a CPU-only machine in an acceptable time. Ie. on the order of 25
  min for 4000 steps.
- Make it runnable locally without kube, Slurm or similar.
- Add snapshotting of configuration and code state for reproducibility.
- Add overseer error injection.
- Add measuring of asker accuracy.
- Add layout definitions for the TensorBoard Custom Scalars plugin.
- Add `managed_experiment.py`, which sends you an email when a run fails. –
  Configuration format undocumented.
