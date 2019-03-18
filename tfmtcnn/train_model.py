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
r"""Train a model using either of PNet, RNet or ONet.

Usage:
```shell

$ python tfmtcnn/tfmtcnn/train_model.py \
    --network_name PNet \
    --train_root_dir ../data/models/mtcnn/train \
    --dataset_root_dir ../data/datasets/mtcnn \
    --base_learning_rate 0.001 \
    --max_number_of_epoch 19 \
    --test_dataset FDDBDataset \
    --test_annotation_image_dir /datasets/FDDB/ \
    --test_annotation_file /datasets/FDDB/FDDB-folds/FDDB-fold-01-ellipseList.txt

$ python tfmtcnn/tfmtcnn/train_model.py \
    --network_name RNet \
    --train_root_dir ../data/models/mtcnn/train \
    --dataset_root_dir ../data/datasets/mtcnn \
    --base_learning_rate 0.001 \
    --max_number_of_epoch 22 \
    --test_dataset FDDBDataset \
    --test_annotation_image_dir /datasets/FDDB/ \
    --test_annotation_file /datasets/FDDB/FDDB-folds/FDDB-fold-01-ellipseList.txt

$ python tfmtcnn/tfmtcnn/train_model.py \
    --network_name ONet \
    --train_root_dir ../data/models/mtcnn/train \
    --dataset_root_dir ../data/datasets/mtcnn \
    --base_learning_rate 0.001 \
    --max_number_of_epoch 21 \
    --test_dataset FDDBDataset \
    --test_annotation_image_dir /datasets/FDDB/ \
    --test_annotation_file /datasets/FDDB/FDDB-folds/FDDB-fold-01-ellipseList.txt
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import argparse

from tfmtcnn.trainers.SimpleNetworkTrainer import SimpleNetworkTrainer
from tfmtcnn.trainers.HardNetworkTrainer import HardNetworkTrainer

from tfmtcnn.networks.NetworkFactory import NetworkFactory


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--network_name',
        type=str,
        help='The name of the network.',
        choices=['PNet', 'RNet', 'ONet'],
        default='PNet')

    parser.add_argument(
        '--dataset_root_dir',
        type=str,
        help='The directory where the dataset files are stored.',
        default=None)

    parser.add_argument(
        '--train_root_dir',
        type=str,
        help='Input train root directory where model weights are saved.',
        default=None)

    parser.add_argument(
        '--base_learning_rate',
        type=float,
        help='Initial learning rate.',
        default=0.001)

    parser.add_argument(
        '--max_number_of_epoch',
        type=int,
        help='The maximum number of training epoch.',
        default=30)

    parser.add_argument(
        '--log_every_n_steps',
        type=int,
        help='The frequency with which logs are printed.',
        default=1000)

    parser.add_argument(
        '--test_dataset',
        type=str,
        help='Test dataset name.',
        choices=['WIDERFaceDataset', 'CelebADataset', 'FDDBDataset'],
        default='FDDBDataset')

    parser.add_argument(
        '--test_annotation_file',
        type=str,
        help='Face dataset annotations file used for evaluating the trained model.',
        default=None)

    parser.add_argument(
        '--test_annotation_image_dir',
        type=str,
        help='Face dataset image directory used for evaluating the trained model.',
        default=None)

    parser.add_argument(
        '--batch_size',
        type=int,
        help='Number of images to process in a batch.',
        default=SimpleNetworkTrainer.default_batch_size())

    return parser.parse_args(argv)


def main(args):
    if not (args.network_name in ['PNet', 'RNet', 'ONet']):
        raise ValueError(f"The network name should be either PNet, RNet or ONet.")

    if not args.dataset_root_dir:
        raise ValueError(f"You must supply the input dataset directory with --dataset_root_dir.")

    if not args.test_annotation_file:
        raise ValueError(f"You must supply face dataset annotations file used for evaluating the "
                         f"trained model with --test_annotation_file.")
    if not args.test_annotation_image_dir:
        raise ValueError(f"You must supply face dataset image directory used for evaluating the trained "
                         f"model with --test_annotation_image_dir.")

    if args.network_name == 'PNet':
        trainer = SimpleNetworkTrainer(args.network_name, batch_size=args.batch_size)
    else:
        trainer = HardNetworkTrainer(args.network_name, batch_size=args.batch_size)

    status = trainer.load_test_dataset(args.test_dataset, args.test_annotation_image_dir, args.test_annotation_file)
    if not status:
        print('Error loading the test dataset for evaluating the trained model.')

    if args.train_root_dir:
        train_root_dir = args.train_root_dir
    else:
        train_root_dir = NetworkFactory.model_train_dir()

    status = trainer.train(args.network_name, args.dataset_root_dir,
                           train_root_dir, args.base_learning_rate,
                           args.max_number_of_epoch, args.log_every_n_steps)
    if status:
        print(f"{args.network_name} - network is trained and weights are generated at {train_root_dir}.")
    else:
        # print('Error training the model.')
        raise RuntimeError('Error training the model.')


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main(parse_arguments(sys.argv[1:]))
