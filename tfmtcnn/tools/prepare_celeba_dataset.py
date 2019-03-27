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
r"""Prepare CelebA dataset for input.

Usage:
```shell

$ python tfmtcnn/tfmtcnn/tools/prepare_celeba_dataset.py \
    --bounding_box_file_name=../data/CelebA/list_bbox_celeba.txt \
    --landmark_file_name=../data/CelebA/list_landmarks_celeba.txt \
    --output_file_name=../data/CelebA/CelebA.txt 
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import re
import argparse

import mef


def prepare_dataset(bounding_box_file_name, landmark_file_name,
                    output_file_name):
    if not (mef.isfile(bounding_box_file_name) and
            mef.isfile(landmark_file_name)):
        return False

    bounding_box_file = open(bounding_box_file_name, 'r')
    landmark_file = open(landmark_file_name, 'r')
    output_file = open(output_file_name, 'w')

    number_of_bounding_boxes = int(bounding_box_file.readline())
    number_of_landmarks = int(landmark_file.readline())

    if number_of_bounding_boxes != number_of_landmarks:
        return False

    # Read bounding box file header.
    bounding_box_file.readline()

    # Read landmark file header.
    landmark_file.readline()

    bounding_boxes = bounding_box_file.readlines()
    landmarks = landmark_file.readlines()

    if number_of_bounding_boxes != number_of_landmarks or \
            len(bounding_boxes) != len(landmarks) or \
            len(bounding_boxes) != number_of_bounding_boxes or \
            number_of_bounding_boxes <= 0:
        return False

    print(f"Extracting info and writing to {output_file_name}...")
    pt = mef.ProgressText(len(bounding_boxes))

    for bounding_box, landmark in zip(bounding_boxes, landmarks):
        bounding_box_info = bounding_box.strip('\n').strip('\r')
        # noinspection PyPep8
        bounding_box_info = re.sub('\s+', ' ', bounding_box_info)
        bounding_box_info = bounding_box_info.split(' ')

        landmark_info = landmark.strip('\n').strip('\r')
        # noinspection PyPep8
        landmark_info = re.sub('\s+', ' ', landmark_info)
        landmark_info = landmark_info.split(' ')

        bounding_box_current_file_name = bounding_box_info[0]
        landmark_current_file_name = landmark_info[0]

        if bounding_box_current_file_name != landmark_current_file_name or \
                len(bounding_box_info) != 5 or len(landmark_info) != 11:
            continue

        output_file.write(
            landmark_current_file_name +
            ' %d %d %d %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n'
            % (int(bounding_box_info[1]), int(bounding_box_info[2]),
               int(bounding_box_info[3]), int(bounding_box_info[4]),
               float(landmark_info[1]), float(landmark_info[2]),
               float(landmark_info[3]), float(landmark_info[4]),
               float(landmark_info[5]), float(landmark_info[6]),
               float(landmark_info[7]), float(landmark_info[8]),
               float(landmark_info[9]), float(landmark_info[10])))
        pt.update()
    return True


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--bounding_box_file_name',
        type=str,
        help='Input CelebA dataset bounding box file name.',
        default=None)

    parser.add_argument(
        '--landmark_file_name',
        type=str,
        help='Input CelebA dataset landmark file name.',
        default=None)

    parser.add_argument(
        '--output_file_name',
        type=str,
        help='Output file name where CelebA dataset file is saved.',
        default=None)

    return parser.parse_args(argv)


def main(args):

    if not args.bounding_box_file_name:
        raise ValueError(
            'You must supply input CelebA dataset bounding box file name with --bounding_box_file_name.'
        )

    if not args.landmark_file_name:
        raise ValueError(
            'You must supply input CelebA dataset landmark file name with --landmark_file_name.'
        )

    if not args.output_file_name:
        raise ValueError(
            'You must supply output file name for storing CelebA dataset file with --output_file_name.'
        )

    status = prepare_dataset(args.bounding_box_file_name, args.landmark_file_name, args.output_file_name)
    if status:
        print('CelebA dataset is generated at ' + args.output_file_name)
    else:
        print('Error generating CelebA dataset.')


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
