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

class WIDERFaceDataset(object):

	def __init__(self, name='WIDERFaceDataset'):
		self._name = name
		self._clear()

	def _clear(self):
		self._is_valid = False
		self._data = dict()
		
	def is_valid(self):
		return(self._is_valid)

	def data(self):
		return(self._data)

	def read(self, annotation_image_dir, annotation_file_name):
		
		if(not os.path.isfile(annotation_file_name)):
			return(False)

		self._clear()

		images = []
		bounding_boxes = []
		annotation_file = open(annotation_file_name, 'r')
		while( True ):
       			image_path = annotation_file.readline().strip('\n')
       			if( not image_path ):
       				break			

       			image_path = os.path.join(annotation_image_dir, image_path)
			image = cv2.imread(image_path)
			if(image is None):
				continue

       			images.append(image_path)

       			nums = annotation_file.readline().strip('\n')
       			one_image_boxes = []
       			for face_index in range(int(nums)):
       				bounding_box_info = annotation_file.readline().strip('\n').split(' ')

       				face_box = [float(bounding_box_info[i]) for i in range(4)]

       				xmin = face_box[0]
       				ymin = face_box[1]
       				xmax = xmin + face_box[2]
       				ymax = ymin + face_box[3]

       				one_image_boxes.append([xmin, ymin, xmax, ymax])

       			bounding_boxes.append(one_image_boxes)

		if(len(images)):			
			self._data['images'] = images
			self._data['bboxes'] = bounding_boxes
			self._is_valid = True
		else:
			self._clear()

		return(self.is_valid())
