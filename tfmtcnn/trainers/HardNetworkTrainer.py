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

import numpy as np
import tensorflow as tf

from tfmtcnn.trainers.SimpleNetworkTrainer import SimpleNetworkTrainer
from tfmtcnn.datasets.TensorFlowDataset import TensorFlowDataset

import mef
import mef_tf


class HardNetworkTrainer(SimpleNetworkTrainer):
    def __init__(self, network_name='RNet', batch_size=-1):
        SimpleNetworkTrainer.__init__(self, network_name, batch_size=batch_size)

    def _read_data(self, dataset_root_dir):
        dataset_dir = self.dataset_dir(dataset_root_dir)

        positive_file_name = self._positive_file_name(dataset_dir)
        part_file_name = self._part_file_name(dataset_dir)
        negative_file_name = self._negative_file_name(dataset_dir)
        image_list_file_name = self._image_list_file_name(dataset_dir)

        tensorflow_file_names = [
            positive_file_name, part_file_name, negative_file_name,
            image_list_file_name
        ]

        positive_ratio = 1.0 / 6
        part_ratio = 1.0 / 6
        landmark_ratio = 1.0 / 6
        negative_ratio = 3.0 / 6

        positive_batch_size = int(np.ceil(self._batch_size * positive_ratio))
        part_batch_size = int(np.ceil(self._batch_size * part_ratio))
        negative_batch_size = int(np.ceil(self._batch_size * negative_ratio))
        landmark_batch_size = int(np.ceil(self._batch_size * landmark_ratio))

        batch_sizes = [
            positive_batch_size, part_batch_size, negative_batch_size,
            landmark_batch_size
        ]

        self._number_of_samples = 0
        for d in tensorflow_file_names:
            # self._number_of_samples += sum(1 for _ in tf.python_io.tf_record_iterator(d))
            mef.tsprint(f"Obtaining number of samples from TFRecords file {d}...")
            n = mef_tf.get_num_tfrecords(d, show_progress=1000)
            mef.tsprint(f"{n} records in TFRecords file.")
            self._number_of_samples += n

        image_size = self.network_size()
        tensorflow_dataset = TensorFlowDataset()
        return tensorflow_dataset.read_tensorflow_files(tensorflow_file_names, batch_sizes, image_size)
