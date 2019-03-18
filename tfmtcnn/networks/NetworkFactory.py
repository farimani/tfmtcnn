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

from tfmtcnn.networks.PNet import PNet
from tfmtcnn.networks.RNet import RNet
from tfmtcnn.networks.ONet import ONet


class NetworkFactory(object):
    def __init__(self):
        pass

    @classmethod
    def network(cls, network_name='PNet'):
        if network_name == 'PNet':
            network_object = PNet()
            return network_object
        elif network_name == 'RNet':
            network_object = RNet()
            return network_object
        elif network_name == 'ONet':
            network_object = ONet()
            return network_object
        else:
            network_object = PNet()
            return network_object

    @classmethod
    def network_size(cls, network_name='PNet'):
        if network_name == 'PNet':
            network_size = 12
        elif network_name == 'RNet':
            network_size = 24
        elif network_name == 'ONet':
            network_size = 48
        else:
            network_size = 12

        return network_size

    @classmethod
    def previous_network(cls, network_name='PNet'):
        if network_name == 'ONet':
            previous_network = 'RNet'
        elif network_name == 'RNet':
            previous_network = 'PNet'
        else:
            previous_network = 'PNet'

        return previous_network

    @classmethod
    def model_deploy_dir(cls):
        model_root_dir, _ = os.path.split(os.path.realpath(__file__))
        model_root_dir = os.path.join(model_root_dir, '../models/mtcnn/deploy/')
        return model_root_dir

    @classmethod
    def model_train_dir(cls):
        model_root_dir, _ = os.path.split(os.path.realpath(__file__))
        model_root_dir = os.path.join(model_root_dir, '../../data/models/mtcnn/train/')
        return model_root_dir

    @classmethod
    def loss_ratio(cls, network_name):
        if network_name == 'PNet':
            class_loss_ratio = 1.0
            bbox_loss_ratio = 0.5
            landmark_loss_ratio = 0.5
        elif network_name == 'RNet':
            class_loss_ratio = 1.0
            bbox_loss_ratio = 0.5
            landmark_loss_ratio = 0.5
        elif network_name == 'ONet':
            class_loss_ratio = 1.0
            bbox_loss_ratio = 0.5
            landmark_loss_ratio = 1.0
        else:  # PNet
            class_loss_ratio = 1.0
            bbox_loss_ratio = 0.5
            landmark_loss_ratio = 0.5

        return class_loss_ratio, bbox_loss_ratio, landmark_loss_ratio
