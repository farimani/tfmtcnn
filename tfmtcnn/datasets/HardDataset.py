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
import numpy as np
import cv2

from tfmtcnn.datasets.SimpleDataset import SimpleDataset
from tfmtcnn.datasets.SimpleFaceDataset import SimpleFaceDataset
from tfmtcnn.datasets.HardFaceDataset import HardFaceDataset
from tfmtcnn.datasets.LandmarkDataset import LandmarkDataset
from tfmtcnn.datasets.TensorFlowDataset import TensorFlowDataset

import tfmtcnn.datasets.constants as datasets_constants

from tfmtcnn.networks.FaceDetector import FaceDetector
from tfmtcnn.networks.NetworkFactory import NetworkFactory


class HardDataset(SimpleDataset):
    def __init__(self, name):
        SimpleDataset.__init__(self, name)
        self._minimum_face_size = datasets_constants.minimum_face_size

    def _generate_image_samples(self, annotation_file_name,
                                annotation_image_dir, model_train_dir,
                                base_number_of_images, target_root_dir):
        status = True

        hard_face_dataset = HardFaceDataset()
        status = hard_face_dataset.generate_samples(
            annotation_image_dir, annotation_file_name, model_train_dir,
            self.network_name(), self._minimum_face_size,
            target_root_dir) and status

        simple_face_dataset = SimpleFaceDataset()
        status = simple_face_dataset.generate_samples(
            annotation_image_dir, annotation_file_name, base_number_of_images,
            NetworkFactory.network_size(self.network_name()),
            target_root_dir) and status

        return (status)

    def _generate_dataset(self, target_root_dir):
        tensorflow_dataset = TensorFlowDataset()

        print('Generating TensorFlow dataset for positive images.')
        if (not tensorflow_dataset.generate(
                SimpleFaceDataset.positive_file_name(target_root_dir),
                target_root_dir, 'positive')):
            print('Error generating TensorFlow dataset for positive images.')
            return (False)
        print('Generated TensorFlow dataset for positive images.')

        print('Generating TensorFlow dataset for partial images.')
        if (not tensorflow_dataset.generate(
                SimpleFaceDataset.part_file_name(target_root_dir),
                target_root_dir, 'part')):
            print('Error generating TensorFlow dataset for partial images.')
            return (False)
        print('Generated TensorFlow dataset for partial images.')

        print('Generating TensorFlow dataset for negative images.')
        if (not tensorflow_dataset.generate(
                SimpleFaceDataset.negative_file_name(target_root_dir),
                target_root_dir, 'negative')):
            print('Error generating TensorFlow dataset for negative images.')
            return (False)
        print('Generated TensorFlow dataset for negative images.')

        print('Generating TensorFlow dataset for landmark images.')
        if (not tensorflow_dataset.generate(
                self._image_list_file_name(target_root_dir), target_root_dir,
                'image_list')):
            print('Error generating TensorFlow dataset for landmark images.')
            return (False)
        print('Generated TensorFlow dataset for landmark images.')

        return (True)

    def generate(self, annotation_image_dir, annotation_file_name,
                 landmark_image_dir, landmark_file_name, model_train_dir,
                 base_number_of_images, target_root_dir):

        if (not os.path.isfile(annotation_file_name)):
            return (False)
        if (not os.path.exists(annotation_image_dir)):
            return (False)

        if (not os.path.isfile(landmark_file_name)):
            return (False)
        if (not os.path.exists(landmark_image_dir)):
            return (False)

        target_root_dir = os.path.expanduser(target_root_dir)
        target_root_dir = os.path.join(target_root_dir, self.network_name())
        if (not os.path.exists(target_root_dir)):
            os.makedirs(target_root_dir)

        self._minimum_face_size = datasets_constants.minimum_face_size

        print('Generating image samples.')
        status = self._generate_image_samples(
            annotation_file_name, annotation_image_dir, model_train_dir,
            base_number_of_images, target_root_dir)
        if (not status):
            print('Error generating image samples.')
            return (False)
        print('Generated image samples.')

        print('Generating landmark samples.')
        if (not super(HardDataset, self)._generate_landmark_samples(
                landmark_image_dir, landmark_file_name, base_number_of_images,
                target_root_dir)):
            print('Error generating landmark samples.')
            return (False)
        print('Generated landmark samples.')

        if (not self._generate_image_list(target_root_dir)):
            return (False)

        print('Generating TensorFlow dataset.')
        if (not self._generate_dataset(target_root_dir)):
            print('Error generating TensorFlow dataset.')
            return (False)
        print('Generated TensorFlow dataset.')

        return (True)
