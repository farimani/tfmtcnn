#######################################################################################
#
# mef_tf.py
# Project General
#
# Mehran's handy tensorflow helpers.
#
# Created by mehran on 01 / 30 / 19.
# Copyright © 2019 Percipo Inc. All rights reserved.
#
#######################################################################################

import os
import mef
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile


def latest_checkpoint_file(model_dir):
    """
    Return the latest checkpoint file in the model training directory.

    :param model_dir:
    :return: The basename for the file, or empty string
    """
    # MEF: Following doesnt work when you move model dirs around, so we need to break it down
    # ckpt_file = tf.train.latest_checkpoint(model_dir)
    ckpt = tf.train.get_checkpoint_state(model_dir)

    if ckpt and ckpt.model_checkpoint_path:
        return mef.basename(ckpt.model_checkpoint_path)

    return ""


def get_model_filenames(model_dir_or_prefix):
    """
    Return the metagraph filename and the latest check point file prefix in the directory

    :param model_dir_or_prefix: Either a path to the model directory, or a sepcific checkpoint prefix as you'd pass to
           saver.restore(), e.g. something like '20190306-205849/model-20190306-205849.ckpt-7'
    :return: basenames of the metafile and last (or the specific, if argument is a ckpt prefix) checkpoint file prefix.
    """

    is_dir = mef.path_type(model_dir_or_prefix) == "dir"
    model_dir = model_dir_or_prefix if is_dir else mef.dirname(model_dir_or_prefix)
    meta_files = mef.scan_dir(model_dir, ".meta", absolute=False, recursive=False)

    if len(meta_files) == 0:
        raise ValueError(f"No meta file found in the model directory ({model_dir})")
    elif len(meta_files) > 1:
        raise ValueError(f"There are {len(meta_files)} meta files in the model directory instead of just one.")

    ckpt_prefix = latest_checkpoint_file(model_dir) if is_dir else mef.basename(model_dir_or_prefix)

    if ckpt_prefix == "":
        raise ValueError(f"There are no checkpoint files in the model directory {model_dir}.")

    return mef.basename(meta_files[0]), ckpt_prefix


def load_model(model_path, input_map=None, sess=None, print_func=print):
    """
    modified from dave sandberg's facenet.

    Check if the model_path is a model directory (containing a metagraph and a checkpoint file), a checkpoint prefix,
    or if it is a protobuf file with a frozen graph. Then load the model accordingly.

    :param model_path:
    :param input_map:
    :param sess: Session to use for checkpoint restoration. If none, default session is used.
    :param print_func: function to print msgs. pass mef.noop() for no msgs or mef.tsprint() for timestamped.
    :return: name of file model was loaded from. if model_path is a directory or a ckpt prefix, then this is
             the path to the checkpoint prefix.
    """

    model_path = os.path.expanduser(model_path)
    path_type = mef.path_type(model_path)

    if path_type == "file":
        print_func(f"Model filename: {model_path}")
        with tf.gfile.GFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, input_map=input_map, name='')

        ret = model_path
    else:
        meta_file, ckpt_file_prefix = get_model_filenames(model_path)
        model_dir = model_path if path_type == "dir" else mef.dirname(model_path)
        print_func(f"Model directory        : {model_dir}")
        print_func(f"Metagraph file         : {meta_file}")
        print_func(f"Checkpoint file prefix : {ckpt_file_prefix}")

        sess = tf.get_default_session() if sess is None else sess
        if sess is None:
            raise RuntimeError("There must be an active Tensorflow session to load a model from checkpoint.")

        saver = tf.train.import_meta_graph(os.path.join(model_dir, meta_file), input_map=input_map)
        ckpt_file_prefix = os.path.join(model_dir, ckpt_file_prefix)
        saver.restore(sess, ckpt_file_prefix)
        ret = ckpt_file_prefix

    return ret


def get_graph_ops(graph):
    """
    Return list of all operations in the graph

    :param graph: a tf.Graph() or tf.GraphDef()
    :return: List of tf.
    """
    return graph.get_operations()


def get_graph_variables(graph):
    """
    Return a list of variable ops in the graph.

    :param graph:
    :return:
    """
    variables = []
    for op in graph.get_operations():
        if op.type.startswith("Variable") or op.type.endswith("VariableOp"):
            variables.append(op)

    return variables


def get_graph_def_nodes(gdef):
    """
    Return list of all operations in the graph def as node defs.

    :param gdef:
    :return:
    """
    return gdef.node


def get_graph_def_variables(gdef):
    """
    Return a list of variable nodedefs in the graph def.

    :param gdef:
    :return:
    """
    variables = []
    for node in gdef.node:
        if node.op.startswith("Variable") or node.op.endswith("VariableOp"):
            variables.append(node)

    return variables


def print_trainable_parameters():
    """
    Print out the trainanble parameters in the current graph
    :return:
    """
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        print(shape)
        print(len(shape))
        variable_parameters = 1
        for dim in shape:
            print(dim)
            variable_parameters *= dim.value
        print(variable_parameters)
        total_parameters += variable_parameters
    print(total_parameters)


def convert_graphdef_to_constant(sess, gdef, output_node_names,
                                 variable_names_whitelist=None, variable_names_blacklist=None):
    """

    :param sess:
    :param gdef:
    :param output_node_names: list of output node name strings
    :param variable_names_whitelist:
    :param variable_names_blacklist:
    :return:
    """

    # fix up problematic ops...
    #
    for node in gdef.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in range(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr:
                del node.attr['use_locking']
        elif node.op == 'AssignAdd':
            node.op = 'Add'
            if 'use_locking' in node.attr:
                del node.attr['use_locking']

    # Replace all the variables in the graph with constants of the same values and get a new graph def
    gd = graph_util.convert_variables_to_constants(sess, gdef, output_node_names,
                                                   variable_names_whitelist=variable_names_whitelist,
                                                   variable_names_blacklist=variable_names_blacklist)
    return gd


def write_graph_def(gdef, filename, as_text=False):
    """
    Write a graph def into a file, optionally as a text file.

    :param gdef:
    :param filename:
    :param as_text:
    :return:
    """

    if as_text:
        tf.io.write_graph(gdef, mef.dirname(filename), mef.basename(filename), as_text=True)
    else:
        with gfile.GFile(filename, 'wb') as f:
            f.write(gdef.SerializeToString())
            # print(f"Inference graph written to {filename}.")


def get_num_tfrecords(tfrecords_filename, show_progress=-1, print_func=mef.tsprint):
    """
    Return the number of samples in a tfrecords file...

    :param tfrecords_filename:
    :param show_progress: number of samples to show progress. if <= 0, not shown...
    :param print_func: Print function. if set to mef.noop(), same as if show_progress <= 0
    :return:
    """
    n_samples = 0
    if show_progress > 0 and print_func != mef.noop:
        for _ in tf.python_io.tf_record_iterator(tfrecords_filename):
            if n_samples % show_progress == 0:
                print_func(f"\rNumber of TFRecords: {n_samples}", end='')
            n_samples += 1

        print("")
    else:
        for _ in tf.python_io.tf_record_iterator(tfrecords_filename):
            n_samples += 1

    return n_samples
