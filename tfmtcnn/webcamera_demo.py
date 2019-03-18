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
r"""Webcamera demo.

Usage:
```shell

$ python tfmtcnn/tfmtcnn/webcamera_demo.py

$ python tfmtcnn/tfmtcnn/webcamera_demo.py \
     --webcamera_id 0

$ python tfmtcnn/tfmtcnn/webcamera_demo.py \
     --webcamera_id 0 \
     --threshold 0.9

$ python tfmtcnn/tfmtcnn/webcamera_demo.py \
     --webcamera_id 0 \
     --threshold 0.9 \
     --test_mode

$ python tfmtcnn/tfmtcnn/webcamera_demo.py \
     --webcamera_id 0 \
     --threshold 0.9 \
     --model_root_dir tfmtcnn/models/mtcnn/wider-celeba

$ python tfmtcnn/tfmtcnn/webcamera_demo.py \
     --webcamera_id 0 \
     --threshold 0.9 \
     --minimum_face_size 120 \
     --model_root_dir tfmtcnn/models/mtcnn/wider-celeba
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse

import cv2
import numpy as np

from tfmtcnn.networks.FaceDetector import FaceDetector
from tfmtcnn.networks.NetworkFactory import NetworkFactory


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--webcamera_id', type=int, help='Webcamera ID.', default=0)

    parser.add_argument(
        '--threshold',
        type=float,
        help='Lower threshold value for face probability (0 to 1.0).',
        default=0.8)

    parser.add_argument(
        '--model_root_dir',
        type=str,
        help='Input model root directory where model weights are saved.',
        default=None)

    parser.add_argument('--test_mode', action='store_true')

    parser.add_argument(
        '--minimum_face_size',
        type=int,
        help='Minimum face size.',
        default=100)
    return parser.parse_args(argv)


def main(args):
    if args.model_root_dir:
        model_root_dir = args.model_root_dir
    else:
        if args.test_mode:
            model_root_dir = NetworkFactory.model_train_dir()
        else:
            model_root_dir = NetworkFactory.model_deploy_dir()

    last_network = 'ONet'
    face_detector = FaceDetector(last_network, model_root_dir)
    webcamera = cv2.VideoCapture(args.webcamera_id)
    webcamera.set(3, 600)
    webcamera.set(4, 800)

    while True:
        start_time = cv2.getTickCount()
        status, current_frame = webcamera.read()
        if status:
            input_bgr_image = np.array(current_frame)
            image_height, image_width, image_channels = input_bgr_image.shape
            input_rgb_image = cv2.cvtColor(input_bgr_image, cv2.COLOR_BGR2RGB)

            # boxes_c, landmarks = face_detector.detect(input_bgr_image)
            boxes_c, landmarks = face_detector.detect(input_rgb_image)

            end_time = cv2.getTickCount()
            time_duration = (end_time - start_time) / cv2.getTickFrequency()
            frames_per_sec = 1.0 / time_duration
            cv2.putText(input_bgr_image, '{:.2f} FPS'.format(frames_per_sec),
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),
                        2)

            for index in range(boxes_c.shape[0]):
                bounding_box = boxes_c[index, :4]
                probability = boxes_c[index, 4]
                crop_box = [
                    int(max(bounding_box[0], 0)),
                    int(max(bounding_box[1], 0)),
                    int(min(bounding_box[2], image_width)),
                    int(min(bounding_box[3], image_height))
                ]

                crop_width = crop_box[3] - crop_box[1]
                crop_height = crop_box[2] - crop_box[0]

                if crop_height < args.minimum_face_size or crop_width < args.minimum_face_size:
                    continue

                if probability > args.threshold:
                    cv2.rectangle(input_bgr_image, (crop_box[0], crop_box[1]),
                                  (crop_box[2], crop_box[3]), (0, 255, 0), 1)
                    cv2.putText(input_bgr_image,
                                'Score - {:.2f}'.format(probability),
                                (crop_box[0], crop_box[1] - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("", input_bgr_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print('Error detecting the webcamera.')
            break

    webcamera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main(parse_arguments(sys.argv[1:]))
