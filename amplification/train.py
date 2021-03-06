import time
import threading
from typing import Sequence
import sys
import itertools
from collections import defaultdict

import tensorflow as tf
import numpy as np

import amplification.models as models
from amplification.tasks.core import idk, print_interaction, recursive_run, Task
from amplification.buffer import Buffer
from amplification.logger import Logger

from tensorflow.contrib.memory_stats.python.ops.memory_stats_ops import BytesInUse, MaxBytesInUse, BytesLimit

print_lock = threading.Lock()

# Credits: https://stackoverflow.com/a/48304328/5091738
def recursive_map(f, o):
    if isinstance(o, dict):
        return {k: recursive_map(f, v) for k, v in o.items()}
    elif isinstance(o, list):
        return [recursive_map(f, x) for x in o]
    else:
        return f(o)

# Like Clojure's get-in with /-separated paths.
def multi_access(d, ks):
    for k in ks.split("/"):
        d = d[k]
    return d

def get_accuracy(guesses, ground_truth):
    return np.mean(np.all(guesses == ground_truth, axis=-1))

class Averager():
    def __init__(self):
        self.reset()
    def add(self, c, x, horizon=None):
        with self.locks[c]:
            if horizon is not None:
                self.n[c] *=  (1 - 1/horizon)
                self.sum[c] *= (1 - 1/horizon)
            self.n[c] += 1
            self.sum[c] += x
    def add_all(self, d, horizon=None):
        for c, x in d.items():
            self.add(c, x, horizon)
    def items(self):
        for c in self.n: yield (c, self.get(c))
    def get(self, c):
        with self.locks[c]:
            return self.sum[c] / self.n[c]
    def reset(self):
        self.n = defaultdict(lambda:1e-9)
        self.sum = defaultdict(lambda:0)
        self.locks = defaultdict(threading.Lock)

def get_interactions(run, task, facts, fast_dbs, Qs,
        use_real_answers=False, use_real_questions=True):

    if not use_real_questions:
        assert not use_real_answers
        # Pass questions to Amplify^H'(X). "targets" are the answers to Qs
        # according to H' after interacting with X.
        # Does this also run the whole interaction between H' and X? Yes, I
        # think so, based on asker.py. But as an unrolled loop.
        return  run(["targets", "subQs", "subAs"], facts=facts, Qs=Qs, fast_dbs=fast_dbs)

    if use_real_answers:
        answerer = lambda Qss: np.array([task.answers(Qs, fast_db)
                                         for Qs, fast_db in zip(Qss, fast_dbs)])
    else:
        answerer = lambda Qss: run(['teacher_or_simple'],
                                   facts=facts, Qs=Qss, fast_dbs=fast_dbs,
                                   is_training=False)[0]

    return recursive_run(task, Qs, answerer)

def print_batch(task, Qs, subQs, subAs, As, facts, fast_dbs, **other_As):
    with print_lock:
        print()
        print()
        print("Facts:")
        for fact in facts[0]:
            print("  " + task.repr_fact(fact))
        for i in range(min(5, len(Qs[0]))):
            print()
            print_interaction(task, Qs[0,i], subQs[0,i], subAs[0,i], As[0,i], fast_dbs[0])
            for k, v in sorted(other_As.items()):
                print("{}: {}".format(k, task.repr_answer(v[0,i])))

def log_accuracy(task, Qs, ground_truth, fast_dbs, stats_averager, stepper, **As_by_name):
    classification = np.asarray([[task.classify_question(Q, fast_db) for Q in Q_list]
                                 for fast_db, Q_list in zip(fast_dbs, Qs)])
    total = np.size(classification)
    with print_lock:
        print()
        for s in ["answerer_gen", "answerer_train"]:
            print(s, stepper[s])
        for name, As in As_by_name.items():
            accuracies = np.all(As == ground_truth, axis=-1)
            correct = np.sum(accuracies)
            classes = set(np.reshape(classification, [-1]))
            counts = {c: np.sum(classification == c) for c in classes}
            correct_counts = {c: np.sum((classification == c) * accuracies) for c in classes}
            def repr_accuracy(k, N): return "{}% ({}/{})".format(int(100*k/N), int(k), int(N))
            print()
            print("{} accuracy: {}".format(name, repr_accuracy(correct, total)))
            stats_averager.add("accuracy/{}".format(name), correct/total)
            # This gives the accuracy on the various classes. In the case of
            # permutation powering, the classes are the magnitudes of the
            # powers. Eg. 2/0010 → class 2, 5/0101 → 3, 15/1111 → 4
            for c in sorted(counts.keys()):
                print("  {}: {}".format(c, repr_accuracy(correct_counts[c], counts[c])))
                stats_averager.add("accuracy_on/{}/{}".format(c, name), correct_counts[c]/counts[c])

def inject_errors(task: Task, As_batches: Sequence, fast_dbs, probability: float):
    if probability == 0.0:
        return
    for As, fast_db in zip(As_batches, fast_dbs):
        task.inject_errors(As, fast_db, probability)

def generate_answerer_data(run, task, get_batch, answerer_buffer, stats_averager, stepper,
        use_real_questions=False, use_real_answers=False, error_probability=0.0):
    averager = Averager()
    while True:
        # Sample questions.
        facts, fast_dbs, Qs, ground_truth = get_batch()
        nqs = Qs.shape[1]
        # What are As and teacher_As?
        # ``As`` depends on the use_real_* parameters. But in the setting of
        # CSASupAmp it's the answers Amplify^H'(X) gives.
        As, subQs, subAs = get_interactions(run, task, facts, fast_dbs, Qs,
                use_real_answers=use_real_answers,
                use_real_questions=use_real_questions)
        inject_errors(task, As, fast_dbs, error_probability)
        # This must be the answers directly from X.
        teacher_As, = run(["answerer/teacher/As"],
                                     facts=facts, Qs=Qs, is_training=False)
        # Calculates how close X is to Amplify^H'(X)?
        # The fraction of batches where there are some actual answers (not all
        # idk) and all of X's answers equal those of Amplify^H'(X).
        teacher_quality = np.mean(np.logical_and(
            np.any(teacher_As != idk, axis=-1),
            np.all(teacher_As == As, axis=-1)
        ))
        log_accuracy(task, Qs, ground_truth, fast_dbs, stats_averager, stepper,
                teacher=teacher_As, targets=As)
        print_batch(task, Qs, subQs, subAs, As, facts, fast_dbs,
                    teacher=teacher_As, truth=ground_truth)
        batch = {"facts":facts, "Qs":Qs, "targets":As, "truth":ground_truth}
        answerer_buffer.extend(batch, extendible={x:[1] for x in answerer_buffer.keys()})
        averager.add("quality", teacher_quality, 100)
        stats_averager.add("quality/teacher", teacher_quality)
        if averager.get("quality") > 0.85 and averager.n["quality"] > 50:
            get_batch.difficulty += 1
            averager.reset()
        stepper["answerer_gen"] += 1

def make_validation_buffer(task, instances=1000, nqs=50, min_difficulty=0, max_difficulty=56):
    difficulty_counts = [1] + ([1/(max_difficulty - min_difficulty)] * (max_difficulty - min_difficulty - 1)) + [1]
    difficulty_sum = sum(difficulty_counts)
    difficulty_counts = [int(d * instances / difficulty_sum) for d in difficulty_counts]
    result = Buffer(instances,
        {"facts": [0, task.fact_length],
         "Qs": [0, task.question_length],
         "truth": [0, task.answer_length]},
    )
    for i, n in enumerate(difficulty_counts):
        difficulty = i + min_difficulty
        facts, fast_dbs, Qs, ground_truth = task.get_batch(n, nqs=nqs, difficulty=difficulty)
        batch = {"facts":facts, "Qs":Qs, "truth":ground_truth}
        result.extend(batch, extendible={x:[1] for x in result.keys()})
    return result

def train_answerer(run, answerer_buffer, stats_averager, make_log, stepper, nbatch, task, warmup_time=0):
    validation_buffer = make_validation_buffer(task)
    while not answerer_buffer.has(10*nbatch):
        time.sleep(0.1)
    while True:
        batch = answerer_buffer.sample(nbatch)
        if stepper["answerer_train"] < warmup_time:
            loss, As = run(["answerer/student/loss",
                            "answerer/student/As"],
                            batch, is_training=True)
        else:
            _, loss, As = run(
                [
                    "answerer/train", "answerer/student/loss",
                    "answerer/student/As"
                ],
                batch,
                is_training=True)

        accuracy = get_accuracy(As, batch["truth"])
        stats_averager.add("accuracy/train", accuracy)
        stats_averager.add("loss/answerer", loss)
        stepper["answerer_train"] += 1
        if stepper["answerer_train"] % 5 == 0:
            batch = validation_buffer.sample(nbatch)
            As, = run(["answerer/student/As"], batch, is_training=False)
            accuracy = get_accuracy(As, batch["truth"])
            stats_averager.add("accuracy/validation", accuracy)
        if stepper["answerer_train"] % 10 == 0:
            # This prints four main accuracies. As I understand them:
            # /target is the accuracy of Amplify^H'(Xpa).
            # /teacher is the accuracy of Xpa on root questions/answers.
            # /train is the training accuracy of X on root questions/answers.
            # /validation is the validation accuracy of X on root questions/answers.
            # Xpa is derived from X by Polyak averaging (see CSASupAmp, p. 14),
            # which must be why /teacher lags behind /train.
            make_log()

def generate_asker_data(run, task, get_batch, asker_buffer, stats_averager, stepper,
        use_real_answers=False, max_nbatch=50):
    averager = Averager()
    needed_labels = 10 * max_nbatch
    last_polled = 0
    while True:
        current = stepper["answerer_train"]
        elapsed = current - last_polled
        last_polled = current
        if averager.get("loss") < 0.01:
            rate = 0.01
        elif averager.get("loss") < 0.1:
            rate = 0.1
        else:
            rate = 1
        needed_labels += rate * elapsed
        nbatch = int(min(max_nbatch, needed_labels))
        needed_labels -= nbatch
        if nbatch == 0:
            time.sleep(0.1)
        else:
            facts, fast_dbs, Qs, ground_truth = get_batch(nbatch)
            As, subQs, subAs = get_interactions(run, task, facts, fast_dbs, Qs,
                    use_real_answers=use_real_answers,
                    use_real_questions=True)
            all_transcripts = []
            all_tokens = []
            for batchn in range(nbatch):
                #get on one random question per batch
                #(we throw away the others and pretend they never existed)
                qn = np.random.randint(Qs.shape[1])
                transcripts, tokens = models.asker.make_transcript(Qs[batchn, qn],
                                                                   subQs[batchn, qn],
                                                                   subAs[batchn, qn],
                                                                   As[batchn, qn])
                all_transcripts.append(transcripts)
                all_tokens.append(tokens)
            # transcripts ends up being the ws argument to asker.build
            # (AttentionSequenceModel.build) via the placeholders dictionary in
            # ``train``.
            batch = {"transcripts":np.array(all_transcripts),
                     "token_types":np.array(all_tokens)}
            asker_buffer.extend(batch)

            metric2fetch = {"loss/asker/validation": "asker/loss"}
            # See the comment on similar code in train_asker.
            if (stepper["asker_gen"] // nbatch) % 50 == 0:
                metric2fetch = {**metric2fetch,
                                "accuracy/asker/q/validation": "asker/q_accuracy",
                                "accuracy/asker/a/validation": "asker/a_accuracy",}
            metric2result = run(metric2fetch, batch, is_training=False)
            averager.add("loss", metric2result["loss/asker/validation"], 3000 * rate/nbatch)
            stats_averager.add("loss/asker/validation", averager.get("loss"))

            # Add the metrics that haven't been added yet.
            del metric2result["loss/asker/validation"]
            stats_averager.add_all(metric2result)

            stepper["asker_gen"] += nbatch

def train_asker(run, asker_buffer, stats_averager, stepper, nbatch):
    while not asker_buffer.has(5*nbatch):
        time.sleep(1)
    while True:
        if stepper["asker_train"] > stepper["answerer_train"] + 10000:
            time.sleep(0.1)
        else:
            metric2fetch = {"loss/asker": "asker/loss"}
            # Unlike the other train_*/generate_* methods, this calculates the
            # accuracy only every fifty steps. This means this stat won't always
            # be logged, because the averager values are logged and reset every
            # ten steps.
            if stepper["asker_train"] % 50 == 0:
                metric2fetch = {**metric2fetch,
                                "accuracy/asker/q/train": "asker/q_accuracy",
                                "accuracy/asker/a/train": "asker/a_accuracy",}
            fetches = ["asker/train", metric2fetch]
            batch = asker_buffer.sample(nbatch)

            _, metric2result = run(fetches, batch, is_training=True)
            stats_averager.add_all(metric2result)

            stepper["asker_train"] += 1

def train(task, model, nbatch=50, num_steps=400000,
        path=None,
        stub=False, learn_human_model=True, supervised=False, curriculum=True,
        generation_frequency=10, log_frequency=10,
        buffer_size=10000, asker_data_limit=100000, loss_threshold=0.3,
        warmup_time=0, error_probability=0.0):
    """

    ``stub`` == True means to pass a bunch of zeroes around and not actually do
    anything. See :py:func:`run` in particular.
    """
    if supervised: learn_human_model = False
    if not stub:
        placeholders = {
            "facts": tf.placeholder(tf.int32, [None, None, task.fact_length], name="facts"),
            "Qs": tf.placeholder(tf.int32, [None, None, task.question_length], name="Qs"),
            "targets": tf.placeholder(tf.int32, [None, None, task.answer_length],
                                      name="targets"),
            "transcripts": tf.placeholder(tf.int32, [None, None], name="transcripts"),
            "token_types": tf.placeholder(tf.int32, [None, None], name="token_types"),
            'is_training': tf.placeholder(tf.bool, [], name='is_training'),
        }
    answerer_buffer = Buffer(buffer_size,
        {"facts": [0, task.fact_length],
         "Qs": [0, task.question_length],
         "targets": [0, task.answer_length],
         "truth": [0, task.answer_length]},
        validation_fraction=0.1,
    )
    asker_buffer = Buffer(asker_data_limit,
        {"transcripts":[task.transcript_length],
         "token_types":[task.transcript_length]},
        validation_fraction=0.1,
    )
    #keep track of how many times we've performed each kind of step
    stepper = {"answerer_train":0, "asker_train":0, "answerer_gen":0, "asker_gen":0}

    #this is machinery for passing python objects into sess.run
    #you actually pass in the index of the object in fast_db_communicator,
    #and wrapped python functions use that
    fast_db_communicator = {'next':0}
    fast_db_index = tf.placeholder(tf.int32, [], name="fast_db_index")

    def answer_if_simple_py(fast_db_index, Qs):
        fast_dbs = fast_db_communicator[fast_db_index]
        As = np.zeros(Qs.shape[:-1] + (task.answer_length,), np.int32)
        are_simple = np.zeros(Qs.shape[:-1], np.bool)
        for i, fast_db in enumerate(fast_dbs):
            are_simple[i] = task.are_simple(Qs[i])
            As[i, are_simple[i]] = task.answers(Qs[i, are_simple[i]], fast_db)
        return are_simple, As

    def answer_if_simple_tf(Qs):
        return tf.py_func(answer_if_simple_py, [fast_db_index, Qs], (tf.bool, tf.int32))

    def make_feed(d):
        result = {}
        cleanup = lambda : None
        for k, v in d.items():
            if k in placeholders:
                result[placeholders[k]] = v
            if k == "fast_dbs":
                fast_db_communicator["next"] += 1
                next_index = fast_db_communicator["next"]
                result[fast_db_index] = next_index
                fast_db_communicator[next_index] = v
                def cleanup(): del fast_db_communicator[next_index]
        return result, cleanup

    def run(fetch_names, batch=None, **kwargs):
        if batch is None:
            batch = {}

        kwargs.update(batch)
        if not stub:
            # ops is defined later. Apparently in Python a closure has access to
            # lexical variables that are defined after the closure.
            fetches = recursive_map(lambda fetch_name: multi_access(ops, fetch_name),
                                    fetch_names)
            feed_dict, cleanup = make_feed(kwargs)
            try:
                return sess.run(fetches, feed_dict)
            finally:
                cleanup()
        else:
            def As(batch):
                return np.zeros(batch["Qs"].shape[:2] + (task.answer_length,), dtype=np.int32)
            def subQs(batch):
                return np.zeros(batch["Qs"].shape[:2] +
                            (task.interaction_length, task.question_length), dtype=np.int32)
            def subAs(batch):
                return np.zeros(batch["Qs"].shape[:2] +
                            (task.interaction_length, task.answer_length), dtype=np.int32)
            def loss(batch): return 0.17
            def accuracy(batch): return 0.85
            def train(batch): return None

            stub_impl = {
                "targets": As,
                "subQs":subQs,
                "subAs":subAs,
                "teacher_or_simple": As,
                "answerer":{
                    "train":train,
                    "teacher":{"As":As, "train":train, "loss":loss},
                    "student":{"As":As, "train":train, "loss":loss}
                },
                "asker":{"train":train, "loss":loss, "q_accuracy": accuracy,
                         "a_accuracy": accuracy}
            }
            return recursive_map(
                lambda fetch_name: multi_access(stub_impl, fetch_name)(kwargs),
                fetch_names)

    # The generator and training threads generate stats at every step. (Which I
    # suspect is bad for performance.) The stats_averager averages them until
    # they're logged every tenth step.
    stats_averager = Averager()

    def get_batch(nbatch=nbatch):
        return task.get_batch(nbatch, difficulty=get_batch.difficulty)
    # Yes, you can set arbitrary attributes on a procedure. No, this is not a
    # good idea in most cases, I suspect.
    get_batch.difficulty = 0 if curriculum else float('inf')

    start_time = time.time()
    logger = Logger(log_path=path, step_field="step/answerer_train")
    def make_log():
        log = {}
        log["time"] = time.time() - start_time
        for k, v in stepper.items():
            log["step/{}".format(k)] = float(v)
        if curriculum:
            log["difficulty"] = float(get_batch.difficulty)
        log.update(stats_averager.items())
        stats_averager.reset()
        with print_lock:
            logger.log(log)

    memory_usage = {}
    sess = None
    if not stub:
        ops = model.build(**placeholders, simple_answerer=answer_if_simple_tf)
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        # Credits: https://software.intel.com/en-us/articles/maximize-tensorflow-performance-on-cpu-considerations-and-recommendations-for-inference
        if not tf.test.is_gpu_available():
            config.inter_op_parallelism_threads = 4
            config.intra_op_parallelism_threads = 6

        if tf.test.is_gpu_available():
            with tf.device('/gpu:0'):
                memory_usage['max_bytes_in_use_0'] = MaxBytesInUse()
                memory_usage['bytes_limit_0'] = BytesLimit()
            if not supervised:
                with tf.device('/gpu:1'):
                    memory_usage['max_bytes_in_use_1'] = MaxBytesInUse()
                    memory_usage['bytes_limit_1'] = BytesLimit()

        sess = tf.Session(config=config)
        # from tensorflow.python import debug as tf_debug
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        model.initialize(sess)
        sess.graph.finalize()


    targets = [
        dict(target=train_answerer,
             args=(run, answerer_buffer, stats_averager, make_log, stepper, nbatch, task, warmup_time)),
        dict(target=train_asker,
             args=(run, asker_buffer, stats_averager, stepper, nbatch)),
        dict(target=generate_answerer_data,
             args=(run, task, get_batch, answerer_buffer, stats_averager, stepper),
             kwargs=dict(use_real_questions=not learn_human_model,
                         use_real_answers=supervised,
                         error_probability=error_probability)),
        dict(target=generate_asker_data,
             args=(run, task, get_batch, asker_buffer, stats_averager, stepper)),
    ]

    threads = [threading.Thread(**kwargs) for kwargs in targets]
    for thread in threads:
        if not stub:
            thread.daemon = True
        thread.start()

    while True:
        time.sleep(10)
        # The following doesn't work well when stub == True. – sess is not
        # defined in that case. And the threads are non-daemon threads, so they
        # don't terminate even when the parent thread terminates.
        #
        # Okay, so stub == True really means stub. Ie. apparently it doesn't
        # train anything and just throws around dummy values.
        if stepper["answerer_train"] >= num_steps:
            if memory_usage:
                with print_lock:
                    for name, op in memory_usage.items():
                        print(name, sess.run(op))
            return
