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

from tfmtcnn.networks.AbstractFaceDetector import AbstractFaceDetector
from tfmtcnn.utils.prelu import prelu


class PNet(AbstractFaceDetector):
    def __init__(self):
        AbstractFaceDetector.__init__(self)
        self._network_size = 12
        self._network_name = 'PNet'
        self._input_batch = None
        self._image_width = self._image_height = None
        self._output_bounding_box = None
        self._output_landmarks = None
        self._output_class_probability = -1

    def _setup_basic_network(self, inputs, is_training=True):
        self._end_points = {}

        with slim.arg_scope([slim.conv2d],
                            activation_fn=prelu,
                            weights_initializer=slim.xavier_initializer(),
                            biases_initializer=tf.zeros_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.00005),
                            padding='valid'):

            end_point = 'conv1'
            net = slim.conv2d(inputs, 10, 3, stride=1, scope=end_point)
            self._end_points[end_point] = net

            end_point = 'pool1'
            net = slim.max_pool2d(
                net,
                kernel_size=[2, 2],
                stride=2,
                scope=end_point,
                padding='SAME')
            self._end_points[end_point] = net

            end_point = 'conv2'
            net = slim.conv2d(
                net,
                num_outputs=16,
                kernel_size=[3, 3],
                stride=1,
                scope=end_point)
            self._end_points[end_point] = net

            end_point = 'conv3'
            net = slim.conv2d(
                net,
                num_outputs=32,
                kernel_size=[3, 3],
                stride=1,
                scope=end_point)
            self._end_points[end_point] = net

            end_point = 'conv4_1'
            conv4_1 = slim.conv2d(
                net,
                num_outputs=2,
                kernel_size=[1, 1],
                stride=1,
                scope=end_point,
                activation_fn=tf.nn.softmax)
            self._end_points[end_point] = conv4_1

            end_point = 'conv4_2'
            bounding_box_predictions = slim.conv2d(
                net,
                num_outputs=4,
                kernel_size=[1, 1],
                stride=1,
                scope=end_point,
                activation_fn=None)
            self._end_points[end_point] = bounding_box_predictions

            end_point = 'conv4_3'
            landmark_predictions = slim.conv2d(
                net,
                num_outputs=10,
                kernel_size=[1, 1],
                stride=1,
                scope=end_point,
                activation_fn=None)
            self._end_points[end_point] = landmark_predictions

            return conv4_1, bounding_box_predictions, landmark_predictions

    def setup_training_network(self, inputs):
        convolution_output, bounding_box_predictions, \
            landmark_predictions = self._setup_basic_network(inputs, is_training=True)

        output_class_probability = tf.squeeze(convolution_output, [1, 2], name='class_probability')
        output_bounding_box = tf.squeeze(bounding_box_predictions, [1, 2], name='bounding_box_predictions')
        output_landmarks = tf.squeeze(landmark_predictions, [1, 2], name="landmark_predictions")
        return output_class_probability, output_bounding_box, output_landmarks

    def setup_inference_network(self, checkpoint_path):
        graph = tf.Graph()
        with graph.as_default():
            self._input_batch = tf.placeholder(tf.float32, name='input_batch')
            self._image_width = tf.placeholder(tf.int32, name='image_width')
            self._image_height = tf.placeholder(tf.int32, name='image_height')
            image_reshape = tf.reshape(self._input_batch, [1, self._image_height, self._image_width, 3])

            convolution_output, bounding_box_predictions, \
                landmark_predictions = self._setup_basic_network(image_reshape, is_training=False)

            self._output_class_probability = tf.squeeze(convolution_output, axis=0)
            self._output_bounding_box = tf.squeeze(bounding_box_predictions, axis=0)
            self._output_landmarks = tf.squeeze(landmark_predictions, axis=0)

            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8, allow_growth=True)
            # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
            cfg = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options, log_device_placement=False)
            self._session = tf.Session(config=cfg)
            return self.load_model(self._session, checkpoint_path)

    def detect(self, input_batch):
        image_height, image_width, _ = input_batch.shape
        class_probabilities, bounding_boxes = self._session.run(
            [self._output_class_probability, self._output_bounding_box],
            feed_dict={
                self._input_batch: input_batch,
                self._image_width: image_width,
                self._image_height: image_height
            })
        return class_probabilities, bounding_boxes
