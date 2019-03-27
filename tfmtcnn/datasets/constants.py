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

# MEF: For training set mix, 3:1:1:2 per the allocation used in the paper.
#
negative_parts = 3
positive_parts = 1
partial_parts = 1
landmark_parts = 2

# MEF: Ratio of negatives that have overlap with a bounding box. The ratio that was originally used in the code
#      here comes to about 0.35.
# negatives_from_bbox_ratio = 0.35
negatives_from_bbox_ratio = 0.25

positive_iou = 0.65
part_iou = 0.40
negative_iou = 0.30
abs_negative_iou = 0.1     # MEF: overlap of this much or less means a negative with very little or no face part.

default_base_number_of_images = 700000

minimum_dataset_face_size = 24

minimum_face_size = 20
