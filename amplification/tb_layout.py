"""Layout definitions for the TensorBoard Custom Scalars tab"""

import tensorflow as tf
from tensorboard.summary import v1 as summary_lib
from tensorboard.plugins.custom_scalar import layout_pb2

# Apparently not every run needs a layout file. It's enough to have it somewhere
# below TensorBoards --logdir. If there is more than one layout file,
# TensorBoard shows both layouts or only the older one. I haven't figured out
# what happens when.

# I might have to make multiple charts, comparing only a subset of
# graphs each. For example, {teacher, targets}, {H', targets}.
# Credits: https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/custom_scalar/custom_scalar_demo.py
all_accs_in_one_chart = summary_lib.custom_scalar_pb(
    layout_pb2.Layout(
        category=[
            layout_pb2.Category(
                title="accuracies",
                chart=[
                    layout_pb2.Chart(
                        title="all_accuracies",
                        multiline=layout_pb2.MultilineChartContent(
                            tag=[r"accuracy(_on)?/.*"]
                        ),
                    )
                ],
            )
        ]
    )
)


def write(layout, log_path):
    with tf.summary.FileWriter(log_path) as writer:
        writer.add_summary(layout)
