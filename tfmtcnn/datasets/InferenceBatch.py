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

import cv2
import numpy as np


class InferenceBatch(object):
    def __init__(self, images, batch_size=1, shuffle=False):
        self.images = images
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.size = len(self.images)

        self.current = 0
        self.data = None
        self.label = None

        self.reset()
        self.get_batch()

    def reset(self):
        self.current = 0
        if self.shuffle:
            np.random.shuffle(self.images)

    def has_next(self):
        return self.current + self.batch_size <= self.size

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.has_next():
            self.get_batch()
            self.current += self.batch_size
            return self.data
        else:
            raise StopIteration

    def getindex(self):
        return self.current / self.batch_size

    def getpad(self):
        if self.current + self.batch_size > self.size:
            return self.current + self.batch_size - self.size
        else:
            return 0

    def get_batch(self):
        image_path = self.images[self.current]
        input_bgr_image = cv2.imread(image_path)

        input_rgb_image = cv2.cvtColor(input_bgr_image, cv2.COLOR_BGR2RGB)
        self.data = input_rgb_image
        # self.data = input_bgr_image
