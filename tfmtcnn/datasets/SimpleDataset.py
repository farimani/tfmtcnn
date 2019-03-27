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

from tfmtcnn.datasets.AbstractDataset import AbstractDataset
from tfmtcnn.datasets.LandmarkDataset import LandmarkDataset
from tfmtcnn.datasets.SimpleFaceDataset import SimpleFaceDataset
from tfmtcnn.datasets.TensorFlowDataset import TensorFlowDataset

from tfmtcnn.networks.NetworkFactory import NetworkFactory

import mef


class SimpleDataset(AbstractDataset):
    def __init__(self, network_name='PNet'):
        AbstractDataset.__init__(self, network_name)

    def _generate_landmark_samples(self, landmark_image_dir, landmark_file_name, base_number_of_images,
                                   target_root_dir):
        landmark_dataset = LandmarkDataset()
        return landmark_dataset.generate(landmark_image_dir, landmark_file_name, base_number_of_images,
                                         NetworkFactory.network_size(self.network_name()), target_root_dir)

    def _generate_simple_image_samples(self, annotation_image_dir, annotation_file_name, base_number_of_images,
                                       target_root_dir):
        face_dataset = SimpleFaceDataset()
        return face_dataset.generate_simple_samples(annotation_image_dir, annotation_file_name, base_number_of_images,
                                                    NetworkFactory.network_size(self.network_name()), target_root_dir)

    def _generate_image_list(self, target_root_dir):
        # MEF: Basically a cat of the files below ...
        #
        image_list_file = open(self._image_list_file_name(target_root_dir), 'w')

        files = (("positive", SimpleFaceDataset.positive_file_name(target_root_dir)),
                 ("partial",  SimpleFaceDataset.part_file_name(target_root_dir)),
                 ("negative", SimpleFaceDataset.negative_file_name(target_root_dir)),
                 ("landmark", LandmarkDataset.landmark_file_name(target_root_dir)))
        for desc in files:
            with open(desc[1], 'r') as f:
                lines = f.readlines()
                pt = mef.ProgressText(len(lines), newline_when_done=False)

                # MEF: Skip blank lines (caused by use of linesep, so it shouldn't happen again, but for old files)
                for line in lines:
                    if line.strip():
                        image_list_file.write(line)
                    pt.update(f"Writing list of {desc[0]} samples... ")
        print("")
        return True

    def _generate_tf_dataset(self, target_root_dir):
        tensorflow_dataset = TensorFlowDataset()
        if not tensorflow_dataset.generate(self._image_list_file_name(target_root_dir), target_root_dir, 'image_list'):
            return False

        return True

    def generate_simple(self, annotation_image_dir, annotation_file_name, landmark_image_dir, landmark_file_name,
                        base_number_of_images, target_root_dir):
        if not (mef.isfile(annotation_file_name) and
                mef.isdir(annotation_image_dir) and
                mef.isfile(landmark_file_name) and
                mef.isdir(landmark_image_dir)):
            return False

        target_root_dir = os.path.expanduser(target_root_dir)
        target_root_dir = os.path.join(target_root_dir, self.network_name())
        mef.create_dir_if_necessary(target_root_dir, raise_on_error=True)

        print('Generating image samples.')
        if not self._generate_simple_image_samples(annotation_image_dir, annotation_file_name, base_number_of_images,
                                                   target_root_dir):
            print('Error generating image samples.')
            return False
        print('Generated image samples.')

        print('Generating landmark samples.')
        if not self._generate_landmark_samples(landmark_image_dir, landmark_file_name, base_number_of_images,
                                               target_root_dir):
            print('Error generating landmark samples.')
            return False
        print('Generated landmark samples.')

        print('Generating image lists.')
        if not self._generate_image_list(target_root_dir):
            print('Error generating image lists.')
            return False
        print('Generated image lists.')

        print('Generating TensorFlow dataset.')
        if not self._generate_tf_dataset(target_root_dir):
            print('Error generating TensorFlow dataset.')
            return False
        print('Generated TensorFlow dataset.')

        return True
