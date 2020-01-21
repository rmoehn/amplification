"""Layout definitions for the TensorBoard Custom Scalars tab"""

import os
import glob

import tensorflow as tf
from tensorboard.summary import v1 as summary_lib
from tensorboard.plugins.custom_scalar import layout_pb2


# Turning Python data into protobuf objects ################

type2chart_mapper = {
    "multiline": lambda chart: layout_pb2.Chart(
        title=chart.get("title"),
        multiline=layout_pb2.MultilineChartContent(tag=chart["patterns"]),
    )
}


def py2proto_chart(chart):
    if chart["type"] in type2chart_mapper:
        return type2chart_mapper[chart["type"]](chart)

    raise ValueError("Got chart with unsupported type: {}".format(chart))


def py2proto_cat(category):
    return layout_pb2.Category(
        title=category["title"],
        chart=[py2proto_chart(c) for c in category["charts"]],
        closed=category.get("collapsed", False),
    )


# Credits: https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/custom_scalar/custom_scalar_demo.py
def py2proto(categories):
    return summary_lib.custom_scalar_pb(
        layout_pb2.Layout(category=[py2proto_cat(c) for c in categories])
    )


# Public API ###############################################

# Apparently not every run needs a layout file. It's enough to have it somewhere
# below TensorBoards --logdir. If there is more than one layout file,
# TensorBoard shows both layouts or only the older one. I haven't figured out
# what happens when. Also, one needs to restart the TensorBoard server for every
# layout change.


def write(layout, log_path):
    old_layout_paths = glob.glob(os.path.join(log_path, "events.out.tfevents.*"))
    for p in old_layout_paths:
        os.unlink(p)

    with tf.summary.FileWriter(log_path) as writer:
        writer.add_summary(py2proto(layout))


# Layout definitions #######################################

main_layout = [
    {
        "title": "accuracies pf >= 0",
        "charts": [
            {"type": "multiline", "patterns": [r"accuracy/targets"]},
            {"type": "multiline", "patterns": [r"accuracy/teacher"]},
            {"type": "multiline", "patterns": [r"accuracy_on/[23]/targets"]},
            {"type": "multiline", "patterns": [r"accuracy_on/[23]/teacher"]},
        ],
    },
    {
        "title": "accuracies pf = 0",
        "charts": [
            {"type": "multiline", "patterns": [r"accuracy(_on)?/(?!asker).*"]},
            {"type": "multiline", "patterns": [r"accuracy/asker/[aq]/train"]},
            {"type": "multiline", "patterns": [r"accuracy(_on)?/(?!asker).*targets"]},
            {"type": "multiline", "patterns": [r"accuracy(_on)?/(?!asker).*teacher"]},
            {"type": "multiline", "patterns": [r"accuracy/(?!asker).*"]},
            {
                "type": "multiline",
                "patterns": [r"accuracy_on/2/targets", r"accuracy/asker/q/train"],
            },
        ],
        "collapsed": True,
    },
    {
        "title": "archive",
        "charts": [
            {"type": "multiline", "patterns": [r"accuracy/asker/.*"]},
            {"type": "multiline", "patterns": [r"accuracy/asker/a/.*"]},
            {"type": "multiline", "patterns": [r"accuracy/asker/q/.*"]},
        ],
        "collapsed": True,
    },
]
