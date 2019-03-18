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
r"""Evaluate the MTCNN model for the given dataset.

Usage:
```shell

$ python tfmtcnn/tfmtcnn/evaluate_model.py \
    --annotation_image_dir /datasets/WIDER_Face/WIDER_val/images \
    --annotation_file_name /datasets/WIDER_Face/WIDER_val/wider_face_val_bbx_gt.txt

$ python tfmtcnn/tfmtcnn/evaluate_model.py \
    --model_root_dir tfmtcnn/tfmtcnn/models/mtcnn/wider-celeba \
    --annotation_image_dir /datasets/WIDER_Face/WIDER_val/images \
    --annotation_file_name /datasets/WIDER_Face/WIDER_val/wider_face_val_bbx_gt.txt

$ python tfmtcnn/tfmtcnn/evaluate_model.py \
    --dataset_name FDDBDataset \
    --annotation_image_dir /datasets/FDDB/ \
    --annotation_file_name /datasets/FDDB/FDDB-folds/FDDB-fold-01-ellipseList.txt

$ python tfmtcnn/tfmtcnn/evaluate_model.py \
    --model_root_dir tfmtcnn/tfmtcnn/models/mtcnn/wider-celeba \
    --dataset_name FDDBDataset \
    --annotation_image_dir /datasets/FDDB/ \
    --annotation_file_name /datasets/FDDB/FDDB-folds/FDDB-fold-01-ellipseList.txt
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import argparse

from tfmtcnn.trainers.ModelEvaluator import ModelEvaluator
from tfmtcnn.networks.NetworkFactory import NetworkFactory


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_root_dir',
        type=str,
        help='Input model root directory where model weights are saved.',
        default=None)

    parser.add_argument(
        '--dataset_name',
        type=str,
        help='Input dataset name.',
        choices=['WIDERFaceDataset', 'CelebADataset', 'FDDBDataset'],
        default='WIDERFaceDataset')

    parser.add_argument(
        '--annotation_file_name',
        type=str,
        help='Input face dataset annotations file.',
        default=None)

    parser.add_argument(
        '--annotation_image_dir',
        type=str,
        help='Input face dataset image directory.',
        default=None)

    return parser.parse_args(argv)


def main(args):

    if not args.annotation_file_name:
        raise ValueError('You must supply input face dataset annotations file with --annotation_file_name.')

    if not args.annotation_image_dir:
        raise ValueError('You must supply input face dataset training image directory with --annotation_image_dir.')

    model_evaluator = ModelEvaluator()
    status = model_evaluator.load(args.dataset_name, args.annotation_image_dir, args.annotation_file_name)
    if not status:
        print('Error loading the test dataset.')

    if args.model_root_dir:
        model_root_dir = args.model_root_dir
    else:
        model_root_dir = NetworkFactory.model_deploy_dir()

    last_network = 'ONet'
    status = model_evaluator.create_detector(last_network, model_root_dir)
    if not status:
        print('Error creating the face detector.')

    status = model_evaluator.evaluate(print_result=True)
    if not status:
        print('Error evaluating the model')


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main(parse_arguments(sys.argv[1:]))
