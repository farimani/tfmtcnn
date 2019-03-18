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
# import sys
import random
import cv2
import tensorflow as tf

import mef


def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _convert_to_example_simple(image_example, image_buffer):
    class_label = image_example['label']
    bbox = image_example['bbox']
    roi = [bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']]
    landmark = [
        bbox['xlefteye'], bbox['ylefteye'], bbox['xrighteye'],
        bbox['yrighteye'], bbox['xnose'], bbox['ynose'], bbox['xleftmouth'],
        bbox['yleftmouth'], bbox['xrightmouth'], bbox['yrightmouth']
    ]

    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'image/encoded': _bytes_feature(image_buffer),
                'image/label': _int64_feature(class_label),
                'image/roi': _float_feature(roi),
                'image/landmark': _float_feature(landmark)
            }))
    return example


def _process_image_withoutcoder(filename):
    input_bgr_image = cv2.imread(filename)

    input_rgb_image = cv2.cvtColor(input_bgr_image, cv2.COLOR_BGR2RGB)
    image_data = input_rgb_image.tostring()

    assert (len(input_bgr_image.shape) == 3)

    image_height = input_bgr_image.shape[0]
    image_width = input_bgr_image.shape[1]
    assert (input_bgr_image.shape[2] == 3)

    return image_data, image_height, image_width


class TensorFlowDataset(object):
    # bbox dict with initialized values
    _bbox_dict = {
        'xmin': 0,
        'ymin': 0,
        'xmax': 0,
        'ymax': 0,
        'xlefteye': 0,
        'ylefteye': 0,
        'xrighteye': 0,
        'yrighteye': 0,
        'xnose': 0,
        'ynose': 0,
        'xleftmouth': 0,
        'yleftmouth': 0,
        'xrightmouth': 0,
        'yrightmouth': 0,    
    }

    def __init__(self):
        self._is_valid = False
        self._dataset = []

    def is_valid(self):
        return self._is_valid

    def data(self):
        return self._dataset

    @classmethod
    def tensorflow_file_name(cls, target_dir, target_name):
        target_name = target_name + '.tfrecord'
        file_name = os.path.join(target_dir, target_name)
        return file_name

    def _read_dataset(self, input_file_name):
        self._is_valid = False
        self._dataset = []
        imagelist = open(input_file_name, 'r')

        lines = imagelist.readlines()
        pt = mef.ProgressText(len(lines))

        for i, line in enumerate(lines):
            info = line.strip().split(' ')
            info_len = len(info)

            if info_len > 0:    # MEF: Skip blank lines - problem with legacy code...
                bbox = self._bbox_dict.copy()

                if info_len == 6:
                    bbox['xmin'] = float(info[2])
                    bbox['ymin'] = float(info[3])
                    bbox['xmax'] = float(info[4])
                    bbox['ymax'] = float(info[5])
                elif info_len == 12:
                    bbox['xlefteye'] = float(info[2])
                    bbox['ylefteye'] = float(info[3])
                    bbox['xrighteye'] = float(info[4])
                    bbox['yrighteye'] = float(info[5])
                    bbox['xnose'] = float(info[6])
                    bbox['ynose'] = float(info[7])
                    bbox['xleftmouth'] = float(info[8])
                    bbox['yleftmouth'] = float(info[9])
                    bbox['xrightmouth'] = float(info[10])
                    bbox['yrightmouth'] = float(info[11])
                elif info_len != 2:
                    # positives and partials have 6 fields, negatives have 2 and landmarks have 12
                    raise ValueError(f"Error on line {i+1} from image list file {input_file_name}. "
                                     f"Expected 6 or 12 fields, got {info_len}.")
                data_example = {
                    'filename': info[0],
                    'label': int(info[1]),
                    'bbox': bbox
                }
                self._dataset.append(data_example)

            pt.update(f"Reading from image list {input_file_name}...")

        if len(self._dataset):
            print(f"Shuffling dataset of length {len(self._dataset)}...")
            random.shuffle(self._dataset)
            print(f"Shuffle finished.")
            self._is_valid = True

        return self._is_valid

    @staticmethod
    def _add_to_tfrecord(filename, image_example, tfrecord_writer):
        image_data, height, width = _process_image_withoutcoder(filename)
        example = _convert_to_example_simple(image_example, image_data)
        tfrecord_writer.write(example.SerializeToString())

    @staticmethod
    def read_tensorflow_file(tensorflow_file_name, batch_size, image_size):
        filename_queue = tf.train.string_input_producer([tensorflow_file_name], shuffle=True)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        feat_dict = {
            'image/encoded': tf.FixedLenFeature([], tf.string),  # one image  one record
            'image/label': tf.FixedLenFeature([], tf.int64),
            'image/roi': tf.FixedLenFeature([4], tf.float32),
            'image/landmark': tf.FixedLenFeature([10], tf.float32)
        }
        image_features = tf.parse_single_example(serialized_example, features=feat_dict)
        image = tf.decode_raw(image_features['image/encoded'], tf.uint8)
        image = tf.reshape(image, [image_size, image_size, 3])
        image = (tf.cast(image, tf.float32) - 127.5) / 128.0

        # image = tf.image.per_image_standardization(image)
        label = tf.cast(image_features['image/label'], tf.float32)
        roi = tf.cast(image_features['image/roi'], tf.float32)
        landmark = tf.cast(image_features['image/landmark'], tf.float32)
        image, label, roi, landmark = tf.train.batch([image, label, roi, landmark],
                                                     batch_size=batch_size, num_threads=12, capacity=1 * batch_size)
        label = tf.reshape(label, [batch_size])
        roi = tf.reshape(roi, [batch_size, 4])
        landmark = tf.reshape(landmark, [batch_size, 10])
        return image, label, roi, landmark

    def read_tensorflow_files(self, tensorflow_file_names, batch_sizes, image_size):
        tensorflow_positive_file_name, tensorflow_part_file_name, \
            tensorflow_negative_file_name, tensorflow_landmark_file_name = tensorflow_file_names
        positive_batch_size, part_batch_size, negative_batch_size, landmark_batch_size = batch_sizes

        positive_images, pos_label, pos_roi, pos_landmark = \
            self.read_tensorflow_file(tensorflow_positive_file_name, positive_batch_size, image_size)
        part_images, part_label, part_roi, part_landmark = \
            self.read_tensorflow_file(tensorflow_part_file_name, part_batch_size, image_size)
        neg_image, neg_label, neg_roi, neg_landmark = \
            self.read_tensorflow_file(tensorflow_negative_file_name, negative_batch_size, image_size)
        landmark_image, landmark_label, landmark_roi, landmark_landmark = \
            self.read_tensorflow_file(tensorflow_landmark_file_name, landmark_batch_size, image_size)

        images = tf.concat([positive_images, part_images, neg_image, landmark_image], 0, name="concat/image")
        labels = tf.concat([pos_label, part_label, neg_label, landmark_label], 0, name="concat/label")
        rois = tf.concat([pos_roi, part_roi, neg_roi, landmark_roi], 0, name="concat/roi")
        landmarks = tf.concat([pos_landmark, part_landmark, neg_landmark, landmark_landmark], 0, name="concat/landmark")
        return images, labels, rois, landmarks

    def generate(self, input_file_name, target_root_dir, target_name):
        tensorflow_dir = os.path.join(target_root_dir, 'tensorflow')
        mef.create_dir_if_necessary(tensorflow_dir, raise_on_error=True)

        tensorflow_filename = TensorFlowDataset.tensorflow_file_name(tensorflow_dir, target_name)
        if not self._read_dataset(input_file_name):
            return False

        total_number_of_samples = len(self._dataset)
        number_of_samples = 0
        with tf.python_io.TFRecordWriter(tensorflow_filename) as tfrecord_writer:
            pt = mef.ProgressText(len(self._dataset))

            for i, image_example in enumerate(self._dataset):
                filename = image_example['filename']
                self._add_to_tfrecord(filename, image_example, tfrecord_writer)
                number_of_samples = number_of_samples + 1
                if number_of_samples % 10000 == 0:
                    print(f"\rProcessed {number_of_samples} / {total_number_of_samples} image samples...")

                pt.update()

        return True
