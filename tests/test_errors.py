"""Tests for the code that simulates overseer failure"""

import datetime
import unittest

import hypothesis
import hypothesis.strategies as st
import numpy as np

from amplification import train
from amplification.tasks import iterate


class TestErrors(unittest.TestCase):
    def assertClose(self, first, second, max_difference, msg=None):
        if second == 0.0:
            self.assertEqual(first, 0.0, msg)
        else:
            self.assertLessEqual(abs(second - first), max_difference, msg)

    @hypothesis.given(
        length=st.integers(1, 2),  # length=3 would make it too slow.
        log_iters=st.integers(2, 7),
        nbatch=st.integers(1, 60),
        error_probability=st.floats(0.0, 1.0),
    )
    @hypothesis.settings(
        deadline=datetime.timedelta(milliseconds=1000), max_examples=10
    )
    def test_inject_errors(self, length, log_iters, nbatch, error_probability):
        task = iterate.IterTask(length=length, log_iters=log_iters)
        facts, fast_dbs, Qs, __ = task.get_batch(nbatch)
        As, __, __ = train.get_interactions(
            None,
            task,
            facts,
            fast_dbs,
            Qs,
            use_real_answers=True,
            use_real_questions=True,
        )

        unchanged_As = np.copy(As)
        train.inject_errors(task, As, fast_dbs, error_probability)
        self.assertEqual(As.shape, unchanged_As.shape)
        self.assertClose(
            np.sum(As != unchanged_As) / float(As.size),
            error_probability,
            max_difference=0.2,
        )
