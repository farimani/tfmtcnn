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
import cv2
from math import *

import tfmtcnn.datasets.constants as datasets_constants

import mef

class FDDBDataset(object):

    __name = 'FDDBDataset'
    __minimum_face_size = datasets_constants.minimum_dataset_face_size

    @classmethod
    def name(cls):
        return FDDBDataset.__name

    @classmethod
    def minimum_face_size(cls):
        return FDDBDataset.__minimum_face_size

    def __init__(self):
        self._is_valid = False
        self._data = {}
        self._number_of_faces = 0
        self._clear()

    def _clear(self):
        self._is_valid = False
        self._data = {}
        self._number_of_faces = 0

    def is_valid(self):
        return self._is_valid

    def data(self):
        return self._data

    def number_of_faces(self):
        return self._number_of_faces

    @staticmethod
    def _filter_coordinate(c, m):
        if c < 0:
            return 0
        elif c > m:
            return m
        else:
            return c

    def read(self, annotation_image_dir, annotation_file_name):
        self._clear()
        if not os.path.isfile(annotation_file_name):
            return False

        extension = '.jpg'
        images, bounding_boxes = [], []
        annotation_file = open(annotation_file_name, 'r')

        print(f"Reading FDDB dataset from {annotation_image_dir}...")
        pt = mef.ProgressText(mef.get_line_count(annotation_file_name))

        while True:
            image_path = annotation_file.readline().strip('\n')
            if not image_path:
                break

            image_path = image_path + extension
            image_path = os.path.join(annotation_image_dir, image_path)
            image_width, image_height = mef.get_image_size(image_path)
            if image_width <= 0 or image_height <= 0:
                print(f"WARNING: Could not parse image {image_path}. Skipping...")

            number_of_faces = int(annotation_file.readline().strip('\n'))
            one_image_boxes = []
            for face_index in range(number_of_faces):
                # <major_axis_radius minor_axis_radius angle center_x center_y 1>
                bounding_box_info = annotation_file.readline().strip('\n').split(' ')
                major_axis_radius = float(bounding_box_info[0])
                minor_axis_radius = float(bounding_box_info[1])
                angle = float(bounding_box_info[2])
                center_x = float(bounding_box_info[3])
                center_y = float(bounding_box_info[4])
                tan_t = -1.0 * (minor_axis_radius / major_axis_radius) * tan(angle)
                t = atan(tan_t)
                x1 = center_x + (major_axis_radius * cos(t) * cos(angle) -
                                 minor_axis_radius * sin(t) * sin(angle))
                x2 = center_x + (major_axis_radius * cos(t + pi) * cos(angle) -
                                 minor_axis_radius * sin(t + pi) * sin(angle))
                xmax = self._filter_coordinate(max(x1, x2), image_width)
                xmin = self._filter_coordinate(min(x1, x2), image_width)

                if tan(angle) != 0:
                    tan_t = (minor_axis_radius / major_axis_radius) * (1 / tan(angle))
                else:
                    tan_t = (minor_axis_radius / major_axis_radius) * (1 / (tan(angle) + 0.0001))

                t = atan(tan_t)
                y1 = center_y + (minor_axis_radius * sin(t) * cos(angle) +
                                 major_axis_radius * cos(t) * sin(angle))
                y2 = center_y + (minor_axis_radius * sin(t + pi) * cos(angle) +
                                 major_axis_radius * cos(t + pi) * sin(angle))
                ymax = self._filter_coordinate(max(y1, y2), image_height)
                ymin = self._filter_coordinate(min(y1, y2), image_height)

                xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax - 1), int(ymax - 1)
                width = xmax - xmin + 1
                height = ymax - ymin + 1

                if max(width, height) >= FDDBDataset.minimum_face_size() and width > 0 and height > 0:
                    one_image_boxes.append([xmin, ymin, xmax, ymax])

            pt.update(step=number_of_faces + 2)

            if len(one_image_boxes):
                images.append(image_path)
                bounding_boxes.append(one_image_boxes)
                self._number_of_faces += len(one_image_boxes)

        print(f"Finished reading FDDB dataset from {annotation_image_dir}.")

        if len(images):
            self._data['images'] = images
            self._data['bboxes'] = bounding_boxes
            self._data['number_of_faces'] = self._number_of_faces
            self._is_valid = True
            print(f"{self._number_of_faces} faces in {len(images)} images for FDDB dataset.")
        else:
            self._clear()

        return self.is_valid()
