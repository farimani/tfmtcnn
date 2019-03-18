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

import tensorflow as tf


class AbstractFaceDetector(object):
    def __init__(self):
        self._network_size = 0
        self._network_name = ''
        self._end_points = {}
        self._session = None
        self._is_model_loaded = False
        self._model_path = None

    def network_size(self):
        return self._network_size

    def network_name(self):
        return self._network_name

    def model_path(self):
        return self._model_path

    def _setup_basic_network(self, inputs):
        raise NotImplementedError('Must be implemented by the subclass.')

    def setup_training_network(self, inputs):
        raise NotImplementedError('Must be implemented by the subclass.')

    def setup_inference_network(self, checkpoint_path):
        raise NotImplementedError('Must be implemented by the subclass.')

    def load_model(self, session, checkpoint_path):
        if self._is_model_loaded:
            return True

        if tf.gfile.IsDirectory(checkpoint_path):
            self._model_path = tf.train.latest_checkpoint(checkpoint_path)
        else:
            self._model_path = checkpoint_path

        if not self._model_path:
            self._is_model_loaded = False
        else:
            saver = tf.train.Saver()
            saver.restore(session, self._model_path)
            self._is_model_loaded = True

        return self._is_model_loaded

    def detect(self, data_batch):
        raise NotImplementedError('Must be implemented by the subclass.')
