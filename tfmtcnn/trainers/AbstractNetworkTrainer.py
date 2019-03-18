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

import os

from tfmtcnn.networks.NetworkFactory import NetworkFactory
from tfmtcnn.datasets.TensorFlowDataset import TensorFlowDataset


class AbstractNetworkTrainer(object):
    _default_batch_size = 384

    def __init__(self, network_name, batch_size=-1):
        self._network = NetworkFactory.network(network_name)
        self._number_of_samples = 0
        self._batch_size = self._default_batch_size if batch_size <= 0 else batch_size

    def network_name(self):
        return self._network.network_name()

    def network_size(self):
        return self._network.network_size()

    def batch_size(self):
        return self._batch_size

    @classmethod
    def default_batch_size(cls):
        return cls._default_batch_size

    def dataset_dir(self, dataset_root_dir):
        dataset_dir = os.path.join(dataset_root_dir, self.network_name())
        tensorflow_dir = os.path.join(dataset_dir, 'tensorflow')
        return tensorflow_dir

    def network_train_dir(self, train_root_dir):
        network_train_dir = os.path.join(train_root_dir, self.network_name())
        return network_train_dir

    @staticmethod
    def _positive_file_name(dataset_dir):
        positive_file_name = TensorFlowDataset.tensorflow_file_name(dataset_dir, 'positive')
        return positive_file_name

    @staticmethod
    def _part_file_name(dataset_dir):
        part_file_name = TensorFlowDataset.tensorflow_file_name(dataset_dir, 'part')
        return part_file_name

    @staticmethod
    def _negative_file_name(dataset_dir):
        negative_file_name = TensorFlowDataset.tensorflow_file_name(dataset_dir, 'negative')
        return negative_file_name

    @staticmethod
    def _image_list_file_name(dataset_dir):
        image_list_file_name = TensorFlowDataset.tensorflow_file_name(dataset_dir, 'image_list')
        return image_list_file_name

    def train(self, network_name, dataset_root_dir, train_root_dir,
              base_learning_rate, max_number_of_epoch, log_every_n_steps):
        raise NotImplementedError('Must be implemented by the subclass.')
