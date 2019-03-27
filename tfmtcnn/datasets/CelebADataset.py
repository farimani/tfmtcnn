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
# import cv2
import numpy as np

import tfmtcnn.datasets.constants as datasets_constants
from tfmtcnn.utils.BBox import BBox

import mef


class CelebADataset(object):

    __name = 'CelebADataset'
    __minimum_face_size = datasets_constants.minimum_dataset_face_size

    @classmethod
    def name(cls):
        return CelebADataset.__name

    @classmethod
    def minimum_face_size(cls):
        return CelebADataset.__minimum_face_size

    def __init__(self):
        self._is_valid = False
        self._data = {}
        self._number_of_faces = 0
        self._clear()

    def _clear(self):
        self._is_valid = False
        self._data = dict()
        self._number_of_faces = 0

    def is_valid(self):
        return self._is_valid

    def data(self):
        return self._data

    def read(self, landmark_image_dir, landmark_file_name):
        self._clear()

        if not os.path.isfile(landmark_file_name):
            return False

        images, bounding_boxes, landmarks = [], [], []
        landmark_file = open(landmark_file_name, 'r')
        mef.tsprint(f"Reading images in CelebA landmark file {landmark_file_name}...")
        pt = mef.ProgressText(mef.get_line_count(landmark_file_name))
        lc = 0
        while True:
            line = landmark_file.readline().strip()
            lc += 1
            landmark_data = line.split(' ')
            image_path = landmark_data[0]
            if not image_path:
                break
            else:
                image_path = os.path.join(landmark_image_dir, landmark_data[0])

            bbox_left, bbox_top = int(landmark_data[1]), int(landmark_data[2])
            bbox_width, bbox_height = int(landmark_data[3]), int(landmark_data[4])

            if max(bbox_width, bbox_height) < CelebADataset.minimum_face_size():
                print(f"NOTE: Line {lc} in file {landmark_file_name}: Bounding box too small. Skpping.")
                continue

            bbox = (bbox_left, bbox_top, bbox_left + bbox_width - 1, bbox_top + bbox_height - 1)
            landmark = np.zeros((5, 2))
            for index in range(0, 5):
                landmark_point = (float(landmark_data[5 + 2 * index]), float(landmark_data[5 + 2 * index + 1]))
                landmark[index] = landmark_point

            images.append(image_path)
            bounding_boxes.append(BBox(bbox))
            landmarks.append(landmark)
            self._number_of_faces += 1

            pt.update()

        if len(images):
            self._data['images'] = images
            self._data['bboxes'] = bounding_boxes
            self._data['landmarks'] = landmarks
            self._data['number_of_faces'] = self._number_of_faces
            self._is_valid = True
            print(f"{self._number_of_faces} faces in {len(images)} images for CelebA dataset.")
        else:
            self._clear()

        return self.is_valid()
