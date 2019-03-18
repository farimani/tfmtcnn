# MIT License
#
# Copyright (c) 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from tfmtcnn.networks.RNet import RNet
from tfmtcnn.utils.prelu import prelu


class ONet(RNet):
    def __init__(self, batch_size=1):
        RNet.__init__(self, batch_size)
        self._network_size = 48
        self._network_name = 'ONet'

    def _setup_basic_network(self, inputs, is_training=True):
        self._end_points = {}

        with slim.arg_scope([slim.conv2d],
                            activation_fn=prelu,
                            weights_initializer=slim.xavier_initializer(),
                            biases_initializer=tf.zeros_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.00005),
                            padding='valid'):

            end_point = 'conv1'
            net = slim.conv2d(
                inputs,
                num_outputs=32,
                kernel_size=[3, 3],
                stride=1,
                scope=end_point)
            self._end_points[end_point] = net

            end_point = 'pool1'
            net = slim.max_pool2d(
                net,
                kernel_size=[3, 3],
                stride=2,
                scope=end_point,
                padding='SAME')
            self._end_points[end_point] = net

            end_point = 'conv2'
            net = slim.conv2d(
                net,
                num_outputs=64,
                kernel_size=[3, 3],
                stride=1,
                scope=end_point)
            self._end_points[end_point] = net

            end_point = 'pool2'
            net = slim.max_pool2d(
                net, kernel_size=[3, 3], stride=2, scope=end_point)
            self._end_points[end_point] = net

            end_point = 'conv3'
            net = slim.conv2d(
                net,
                num_outputs=64,
                kernel_size=[3, 3],
                stride=1,
                scope=end_point)
            self._end_points[end_point] = net

            end_point = 'pool3'
            net = slim.max_pool2d(
                net,
                kernel_size=[2, 2],
                stride=2,
                scope=end_point,
                padding='SAME')
            self._end_points[end_point] = net

            end_point = 'conv4'
            net = slim.conv2d(
                net,
                num_outputs=128,
                kernel_size=[2, 2],
                stride=1,
                scope=end_point)
            self._end_points[end_point] = net

            fc_flatten = slim.flatten(net)

            end_point = 'fc1'
            fc1 = slim.fully_connected(
                fc_flatten,
                num_outputs=256,
                scope=end_point,
                activation_fn=prelu)
            self._end_points[end_point] = fc1

            end_point = 'cls_fc'
            class_probability = slim.fully_connected(
                fc1,
                num_outputs=2,
                scope=end_point,
                activation_fn=tf.nn.softmax)
            self._end_points[end_point] = class_probability

            end_point = 'bbox_fc'
            bounding_box_predictions = slim.fully_connected(
                fc1, num_outputs=4, scope=end_point, activation_fn=None)
            self._end_points[end_point] = bounding_box_predictions

            end_point = 'landmark_fc'
            landmark_predictions = slim.fully_connected(
                fc1, num_outputs=10, scope=end_point, activation_fn=None)
            self._end_points[end_point] = landmark_predictions

            return (class_probability, bounding_box_predictions,
                    landmark_predictions)
