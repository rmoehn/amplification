"""
Task is like:

Facts:
Sets of clauses, length 3, each indicates a variable assignment

Questions:
Assignments/wildcard for all variables

Answer:
Number of satisfying assignments of facts+assignment?
"""

import random
from collections import defaultdict

import numpy as np

from core import idk, uniform, Task, sequences, test_task, recursive_run


def matches(patterns, xs):
    # assert len(patterns.shape) == 2
    # assert len(xs.shape) == 2
    if len(xs.shape) < 2:
        return np.zeros((patterns.shape[0], 1))
    try:
        return np.all(
            (patterns[:, np.newaxis, :] == SatTask.wild) |
            (patterns[:, np.newaxis, :] == xs[np.newaxis, :, :]),
            axis=2)
    except IndexError:
        print(patterns)
        print(xs)
        print(patterns.shape, xs.shape)
        import pdb
        pdb.set_trace()


def alternating_sequences(alphabet1, alphabet2, length):
    if length == 0:
        return [[]]
    L = []
    for a in alphabet1:
        for b in alphabet2:
            L.extend([[a, b] + L for L in alternating_sequences(
                alphabet1, alphabet2, length - 1)])
    return L


class SatTask(Task):
    wild = 1
    answer_length = 1
    fixed_vocab = 2

    def repr_symbol(self, x):
        if x == self.wild: return '*'
        if x in self.differences:
            return str(x - self.zero)
        if x in self.variable_names:
            return 'abcdef' [x - self.variable_names[0]]
        if x in self.variable_values:
            return str(x - self.variable_values[0])
        if x == idk: return '?'

        raise ValueError(x)

    def __init__(self, length=3, size=float('inf'), nvars=6, nvalues=2):
        self.nvocab = self.fixed_vocab
        self.nvars = nvars
        self.nvalues = nvalues
        self.size = min(size, nvars**length * nvalues**length)
        # Allocate space for variable names
        self.variable_names = self.allocate(nvars)
        # Allocate space for variable names
        self.variable_values = self.allocate(nvalues)
        # ? maybe number of questions
        self.interaction_length = nvars
        self.variable_values_plus = np.concatenate([[self.wild],
                                                    self.variable_values])
        # Maximum difference
        self.max_d = nvalues**nvars
        # (self.size + nvars - 1) // nvars
        # Allocate vocabulary space for differences
        self.differences = self.allocate(2 * self.max_d + 1)
        self.zero = self.differences[self.max_d]

        self.all_clauses = [
            np.array(x) for x in alternating_sequences(
                self.variable_names, self.variable_values, length)
        ]
        self.length = length
        self.question_length = nvars
        self.fact_length = length * 2

    def satisfying_assignments(self, clauses, assignment=None):
        if assignment is None:
            assignment = []
        if len(assignment) == self.nvars:
            for c in clauses:
                satisfied = False
                for i in range(len(c) // 2):
                    var = c[2 * i] - self.variable_names[0]
                    value = c[2 * i + 1]
                    if assignment[var] == value:
                        satisfied = True
                        break
                if not satisfied:
                    return []
            return [assignment]
        assignments = []
        for value in range(self.nvalues):
            assignments.extend(
                self.satisfying_assignments(
                    clauses, assignment + [value + self.variable_values[0]]))
        return assignments

    def make_dbs(self, difficulty=float('inf')):
        num_clauses = min(self.size, difficulty + 8)
        strings = np.stack(random.sample(self.all_clauses, num_clauses))
        assignments = np.array(self.satisfying_assignments(strings))
        fast_db = {"strings": strings, "assignments": assignments}
        facts = np.concatenate(strings[:, np.newaxis])
        return facts, fast_db

    def answers(self, Qs, fast_db):
        all_assignments = matches(Qs, fast_db["assignments"])
        raw_As = np.sum(all_assignments, axis=1)
        As = self.encode_n(raw_As)
        return As[:, np.newaxis]

    def make_q(self, fast_db):
        Q = np.random.choice(
            self.variable_values, self.question_length, replace=True)
        num_wilds = np.random.randint(1, self.question_length + 1)
        indices = np.random.choice(
            self.question_length, num_wilds, replace=False)
        Q[indices] = self.wild
        return Q

    def encode_n(self, x):
        return self.zero + np.maximum(-self.max_d, np.minimum(self.max_d, x))

    def are_simple(self, Qs):
        return np.all(Qs != self.wild, axis=-1)

    def recursive_answer(self, Q):
        Q = np.asarray(Q)
        if not np.all(np.isin(Q, self.variable_values_plus)):
            yield self.pad(self.zero), None
            return
        if not np.any(Q == self.wild):
            yield (yield None, Q), None
            return
        wild_index = np.argmax(Q == self.wild)
        result = 0
        for c in self.variable_values:
            new_Q = np.copy(Q)
            new_Q[wild_index] = c
            d = (yield None, new_Q)
            if d not in self.differences:
                yield self.pad(idk), None
                return
            result += d - self.zero
        result = self.encode_n(result)
        yield self.pad(result), None

    def all_questions(self, fast_db):
        yield from sequences(self.variable_values_plus, self.nvars)

    def classify_question(self, Q, fast_db):
        n = len([x for x in Q if x == self.wild])
        return "sat{}".format(n)


if __name__ == "__main__":
    # print(alternating_sequences([1, 2], ['a', 'b'], 3))
    task = SatTask(size=2, nvars=2, length=2)
    a = task.variable_names[0]
    b = task.variable_names[1]
    v0 = task.variable_values[0]
    v1 = task.variable_values[1]
    wild = task.wild
    print([
        task.repr_fact(fact) for fact in task.satisfying_assignments(
            np.array([[a, v0, b, v1], [a, v0, b, v0]]))
    ])
    print([
        task.repr_fact(fact) for fact in task.satisfying_assignments(
            np.array([[a, v1, b, v1], [a, v0, b, v0]]))
    ])
    print([
        task.repr_fact(fact) for fact in task.satisfying_assignments(
            np.array([[a, v0, b, v1], [a, v0, b, v0], [a, v1, a, v1]]))
    ])

    task = SatTask(size=2, nvars=3, length=3)
    a = task.variable_names[0]
    b = task.variable_names[1]
    c = task.variable_names[2]
    v0 = task.variable_values[0]
    v1 = task.variable_values[1]
    wild = task.wild
    # ! a = b
    print([
        task.repr_fact(fact) for fact in task.satisfying_assignments(
            np.array([[a, v0, a, v0, b, v0],
                      [a, v1, a, v1, b, v1],
                      [b, v0, b, v0, c, v0],
                      [b, v1, b, v1, c, v1],
                      [a, v0, a, v0, c, v0],
                      [a, v1, a, v1, c, v1],
            ]))
    ])
    print([
        task.repr_fact(fact) for fact in task.satisfying_assignments(
            np.array([[a, v0, a, v0, b, v0],
                      [a, v1, a, v1, b, v1],
                      [b, v0, b, v0, c, v0],
                      [b, v1, b, v1, c, v1],
            ]))
    ])

    task = SatTask()
    nbatch = 1000
    nqs = 1
    fast_dbs = []
    for i in range(nbatch):
        fast_dbs.append(task.make_dbs(difficulty=0)[1])
    # print("\n".join([task.repr_fact(fact) for fact in fast_dbs[0]["strings"]]))
    from collections import defaultdict
    d = defaultdict(lambda : 0)
    for fast_db in fast_dbs:
        answerer = lambda Qs: task.answers(Qs, fast_db)
        Qs = task.make_qs(nqs, fast_db)
        for Q in Qs:
            task.classify_question(Q, fast_db)
        # print(task.repr_question(Qs[0]))
        direct_A = task.answers(Qs, fast_db)
        d[task.repr_answer(direct_A[0])] += 1
        recursive_A, subQs, subAs = recursive_run(task, Qs, answerer)
        assert np.all(direct_A == recursive_A)

    for k in sorted(d.keys()):
        print (k, d[k])

        
    nbatch = 10
    nqs = 300
    import time
    fast_dbs = []
    t0 = time.time()
    for i in range(nbatch):
        fast_dbs.append(task.make_dbs(difficulty=0)[1])
    t = (time.time() - t0) / nbatch
    print("\n".join([task.repr_fact(fact) for fact in fast_dbs[0]["strings"]]))
    print("Generate one environment: {:.1E}".format(t))
    t0 = time.time()
    for fast_db in fast_dbs:
        answerer = lambda Qs: task.answers(Qs, fast_db)
        Qs = task.make_qs(nqs, fast_db)
        for Q in Qs:
            task.classify_question(Q, fast_db)
        print(task.repr_question(Qs[0]))
        direct_A = task.answers(Qs, fast_db)
        print(task.repr_answer(direct_A[0]))
        recursive_A, subQs, subAs = recursive_run(task, Qs, answerer)
        assert np.all(direct_A == recursive_A)
    t = (time.time() - t0) / (nbatch * nqs)
    print("Answer one question: {:.1E}".format(t))
    task.get_batch(10, 10)
