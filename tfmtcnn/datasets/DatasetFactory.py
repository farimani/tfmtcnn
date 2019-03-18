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

import tfmtcnn.datasets.constants as datasets_constants

from tfmtcnn.datasets.WIDERFaceDataset import WIDERFaceDataset
from tfmtcnn.datasets.CelebADataset import CelebADataset
from tfmtcnn.datasets.LFWLandmarkDataset import LFWLandmarkDataset
from tfmtcnn.datasets.FDDBDataset import FDDBDataset


class DatasetFactory(object):

    __positive_iou = datasets_constants.positive_iou
    __part_iou = datasets_constants.part_iou
    __negative_iou = datasets_constants.negative_iou

    def __init__(self):
        pass

    @classmethod
    def positive_iou(cls):
        return DatasetFactory.__positive_iou

    @classmethod
    def part_iou(cls):
        return DatasetFactory.__part_iou

    @classmethod
    def negative_iou(cls):
        return DatasetFactory.__negative_iou

    @classmethod
    def face_dataset(cls, name):
        if name == CelebADataset.name():
            return CelebADataset()
        elif name == WIDERFaceDataset.name():
            return WIDERFaceDataset()
        elif name == FDDBDataset.name():
            return FDDBDataset()
        else:
            return None

    @classmethod
    def landmark_dataset(cls, name):
        if name == CelebADataset.name():
            return CelebADataset()
        elif name == LFWLandmarkDataset.name():
            return LFWLandmarkDataset()
        else:
            return None
