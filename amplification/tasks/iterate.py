import random
from collections import defaultdict

import numpy as np

from amplification.tasks.core import idk, uniform, Task, sequences

class IterTask(Task):
    interaction_length = 3

    zero = 1
    one = 2
    fixed_vocab = 3

    def repr_symbol(self, x):
        if x == idk: return '?'
        if x == self.one: return '1'
        if x == self.zero: return '0'
        if x in self.chars: return 'abcdefghijklmnopqrstuv'[x]

    def __init__(self, nchars=8, length=2, log_iters=6):
        """

        Parameters
        ----------
        log_iters : int
            Base-2 logarithm of the maximum number of iterations/the highest
            power of the permutation. Eg. highest power = 64 → log_iters = 6.
        """
        # Ends up being fixed_vocab + nchars, ie. the total number of symbols used.
        self.nvocab = self.fixed_vocab
        self.nchars = nchars
        # Number of chars to represent one symbol/domain member.
        self.length = length
        self.log_iters = log_iters
        self.size = nchars ** length
        # [fixed_vocab, …, fixed_vocab + nchars - 1]
        # fixed_vocab might better be called first_char.
        # These are just indices of characters, not actual characters.
        self.chars = self.allocate(nchars)
        self.min_char = self.chars[0]
        self.max_char = self.chars[-1]
        # List of all pairs of chars.
        self.vars = list(sequences(self.chars, length))
        # Eg. σ⁵(1) = ? → aa000101 (length 8)
        self.question_length = length + log_iters
        # Eg. σ(1) = 2 → aaab (length 4)
        self.fact_length = 2 * length
        # Eg. 2 → ab (length 2)
        self.answer_length = length

    def make_dbs(self, difficulty=float('inf')):
        # Optionally limit the number of symbols in the permutation.
        size = min(difficulty+8, self.size)
        used_vars = random.sample(self.vars, size)

        # Permutation mapping symbol index to symbol index.
        # 'raw' indicates something returning symbol indices.
        vals_raw = np.random.permutation(size)
        # Permutation mapping symbol index to symbol.
        vals = np.array(used_vars)[vals_raw]

        # 'squares' might better be called 'power_of_2_th_permutations'.
        # No it might not. Permuting permuted values is a squared permutation.
        # Which is clever, but is it necessary? The largest case is batch size *
        # 64 permutations of 64 values. In a compact representation
        # that's a lookup table of batch_size * 4096 bytes. Not mutch.
        # Maybe because the lookup is going to be sparse, it takes more time to
        # precompute all of them than to compute just a few on the fly.
        # squares_raw[k][i] is the 2**k-th permutation of the i-th symbol.
        square_raw = vals_raw
        squares_raw = [square_raw]
        # Would be better to set up the arrays first and then work with indices
        # instead of appending.
        for i in range(self.log_iters):
            square_raw = square_raw[square_raw]
            squares_raw.append(square_raw)
        # Same here.
        # squares[k][x] = σᵏ⁺¹(x)
        squares = [{val: used_vars[squares_raw[i][index]] for index, val in enumerate(used_vars)}
                   for i in range(self.log_iters)]
        fast_db = {
            # Permutation mapping symbol to symbol.
            "vals": {v: val for v, val in zip(used_vars, vals)},
            "vars":  used_vars,
            "squares_raw": squares_raw,
            "squares": squares
        }
        # Permutation mapping symbol to symbol in array form.
        facts = np.concatenate([np.array(used_vars), vals], axis=1)
        return facts, fast_db

    def are_chars(self, x):
        return np.logical_and(np.all(x >= self.min_char), np.all(x <= self.max_char))

    def recursive_answer(self, Q):
        """

        Yields tuples. The first item is the answer, when found, otherwise None.
        The second item is a sub-question, if necessary.
        So the usage is co-routine style. The caller passes a question Q and
        receives a generator. The generator yields sub-questions and expects
        sub-answers in return. Once all sub-questions are answered, it yields
        the overall answer.

        RA = recursive_answer(Q)
              RA → (None, sq1)
        sa1 → RA → (None, sq2)
        sa2 → RA → (A, None)
        """
        Q = tuple(Q)
        x = Q[:self.length]
        # Again, might be better to lift n into binary number space, in order to
        # make the operations involving it more idiomatic.
        n = Q[self.length:]
        if not self.are_chars(x) or not np.all(np.isin(n, [self.zero, self.one])):
            yield self.pad(idk), None
            return
        # The base case? If n < 2…
        if np.all(n[:-1] == self.zero):
            # yield returns None if execution is continued with next(). If it is
            # continued with send(x), yield returns x.
            yield (yield None, Q), None
            return
        leading_bit = np.argmax(n)
        shifted = self.zero * np.ones(self.log_iters, dtype=np.int32)
        shifted[1:] = n[:-1]
        # shifted = n >> 1 if n were a number.
        # 'query' denotes the permutation exponent part of a sub-question.
        queries = [shifted, shifted]
        # If n is odd, we need three sub-questions.
        if n[-1] == self.one:
            parity = self.zero * np.ones(self.log_iters, dtype=np.int32)
            parity[-1] = self.one
            # parity == 1 if parity were a number.
            queries.append(parity)
        def query(x, n): return np.concatenate([x, n])
        for m in queries:
            x = (yield None, query(x, m))
            if not self.are_chars(x):
                yield self.pad(idk), None
                return
        yield self.pad(x), None

    def make_q(self, fast_db):
        # Could be made the last line before then return.
        x = random.choice(fast_db["vars"])

        # Might be better to work with actual binary numbers and only convert to
        # the target format in the end.
        # Randomly draw the position of the leading 1. Then fill the lower
        # places with 0/1 randomly. – But why?
        n = np.ones(self.log_iters, dtype=np.int32) * self.zero
        leading_bit = np.random.randint(0, self.log_iters-1)
        n[leading_bit] = self.one
        remainder = self.log_iters- leading_bit - 1
        n[leading_bit+1:] = np.random.choice([self.zero, self.one], remainder)
        return np.concatenate([x, n])

    def answer(self, Q, fast_db):
        Q = tuple(Q)
        x = Q[:self.length]
        n = Q[self.length:]
        if x in fast_db["vals"]:
            # Compute the n-th permutation in log(n, 2) time by using the
            # precomputed power-of-2-th permutations.
            for i in range(self.log_iters):
                if n[i] == self.one:
                    x = fast_db["squares"][self.log_iters - i - 1][x]
                elif n[i] == self.zero:
                    x = x
                else:
                    return self.pad(idk)
            return self.pad(x)
        else:
            return self.pad(idk)

    def all_questions(self, fast_db):
        for x in fast_db["vars"]:
            for n in sequences([self.zero, self.one], self.log_iters):
                yield np.asarray([x] + n)

    def are_simple(self, Q):
        return np.all(Q[:,self.length:-1] == self.zero, axis=-1)

    def classify_question(self, Q, fast_db):
        n = Q[self.length:]
        if np.all(n == self.zero):
            return 0
        leading_bit = np.argmax(n)
        return self.log_iters - leading_bit
