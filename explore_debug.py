import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

# (For TensorBoard Debugger plugin, might have to modify source_utils.py:
# return False
# # raise ValueError(
# #     "Input file path (%s) is not a Python source file." % py_file_path)
# )

# Does the following work as I want it? Or does it only stop the run at the
# construction phase? I might have to feed a NaN or a specific value.

# Also try out the filtering by regex. Does it work with negative regexes? Does
# it speed up the run? Probably it's slow to run a Python function at every step
# through the graph.

with tf.name_scope("bla"):
    x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = tf.multiply(3, 4, name="f")
breakt = tf.constant(np.nan, name="break")

init = tf.global_variables_initializer()

# ↓↑ Both work. If nans occur in other places, too, I can use the custom filter.

def name_filter(datum, tensor: tf.Tensor):
    print(datum.node_name, tensor)
    return datum.node_name == "break"


# with tf_debug.LocalCLIDebugWrapperSession(tf.Session()) as sess:
with tf.Session() as sess:
    # sess.add_tensor_filter('f_filter', name_filter)
    init.run()
    print(sess.run([x, {'b': y, 'c': f}]))
