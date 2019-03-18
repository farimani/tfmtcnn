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
import numpy as np
import cv2
# import numpy.random as npr

from tfmtcnn.datasets.DatasetFactory import DatasetFactory
from tfmtcnn.datasets.SimpleFaceDataset import SimpleFaceDataset
from tfmtcnn.datasets.InferenceBatch import InferenceBatch
import tfmtcnn.datasets.constants as datasets_constants

from tfmtcnn.networks.FaceDetector import FaceDetector
from tfmtcnn.networks.NetworkFactory import NetworkFactory

from tfmtcnn.utils.convert_to_square import convert_to_square
from tfmtcnn.utils.IoU import iou

import mef


class HardFaceDataset(SimpleFaceDataset):
    def __init__(self, name='HardFaceDataset'):
        SimpleFaceDataset.__init__(self, name)

    @staticmethod
    def _generate_hard_samples(dataset, network_name, detected_boxes, minimum_face_size, target_root_dir):
        image_file_names = dataset['images']
        ground_truth_boxes = dataset['bboxes']
        number_of_images = len(image_file_names)

        if len(detected_boxes) != number_of_images:
            return False

        target_face_size = NetworkFactory.network_size(network_name)
        positive_dir = os.path.join(target_root_dir, 'positive')
        part_dir = os.path.join(target_root_dir, 'part')
        negative_dir = os.path.join(target_root_dir, 'negative')

        mef.create_dir_if_necessary(positive_dir, raise_on_error=True)
        mef.create_dir_if_necessary(part_dir, raise_on_error=True)
        mef.create_dir_if_necessary(negative_dir, raise_on_error=True)

        positive_file = open(SimpleFaceDataset.positive_file_name(target_root_dir), 'w')
        part_file = open(SimpleFaceDataset.part_file_name(target_root_dir), 'w')
        negative_file = open(SimpleFaceDataset.negative_file_name(target_root_dir), 'w')

        generated_negative_samples = generated_positive_samples = generated_part_samples = 0

        mef.tsprint(f"Generating hard sample images...")
        total_images, img_idx = len(image_file_names), 0
        pt = mef.ProgressText(len(image_file_names))

        for image_file_path, detected_box, ground_truth_box in \
                zip(image_file_names, detected_boxes, ground_truth_boxes):
            img_idx += 1
            ground_truth_box = np.array(ground_truth_box, dtype=np.float32).reshape(-1, 4)

            if detected_box.shape[0] == 0:
                pt.update(f"{img_idx} / {total_images}")
                continue

            detected_box = convert_to_square(detected_box)
            detected_box[:, 0:4] = np.round(detected_box[:, 0:4])
            current_image = cv2.imread(image_file_path)
            per_image_face_images = 0

            for box in detected_box:
                x_left, y_top, x_right, y_bottom, _ = box.astype(int)
                width = x_right - x_left + 1
                height = y_bottom - y_top + 1

                if ((width < minimum_face_size) or (height < minimum_face_size)
                        or (x_left < 0) or (y_top < 0)
                        or (x_right > (current_image.shape[1] - 1))
                        or (y_bottom > (current_image.shape[0] - 1))):
                    continue

                current_iou = iou(box, ground_truth_box)
                cropped_image = current_image[y_top:y_bottom + 1, x_left:x_right + 1, :]
                resized_image = cv2.resize(cropped_image, (target_face_size, target_face_size),
                                           interpolation=cv2.INTER_LINEAR)
                if np.max(current_iou) < DatasetFactory.negative_iou():
                    continue
                else:
                    idx = np.argmax(current_iou)
                    assigned_gt = ground_truth_box[idx]
                    x1, y1, x2, y2 = assigned_gt

                    offset_x1 = (x1 - x_left) / float(width)
                    offset_y1 = (y1 - y_top) / float(height)
                    offset_x2 = (x2 - x_right) / float(width)
                    offset_y2 = (y2 - y_bottom) / float(height)

                    if np.max(current_iou) >= DatasetFactory.positive_iou():
                        file_path = os.path.join(positive_dir, f"hard-positive-{generated_positive_samples}.jpg")
                        # MEf: wrong to use linesep here
                        # positive_file.write(file_path + ' 1 %.2f %.2f %.2f %.2f' %
                        #                     (offset_x1, offset_y1, offset_x2, offset_y2) + os.linesep)
                        positive_file.write(f"{file_path} 1 {offset_x1:.2f} {offset_y1:.2f} "
                                            f"{offset_x2:.2f} {offset_y2:.2f}\n")
                        cv2.imwrite(file_path, resized_image)
                        generated_positive_samples += 1
                        per_image_face_images += 1

                    elif np.max(current_iou) >= DatasetFactory.part_iou():
                        file_path = os.path.join(part_dir, f"hard-part-{generated_part_samples}.jpg")
                        # MEf: wrong to use linesep here
                        # part_file.write(file_path + ' -1 %.2f %.2f %.2f %.2f' %
                        #                 (offset_x1, offset_y1, offset_x2, offset_y2) + os.linesep)
                        part_file.write(f"{file_path} 1 {offset_x1:.2f} {offset_y1:.2f} "
                                        f"{offset_x2:.2f} {offset_y2:.2f}\n")
                        cv2.imwrite(file_path, resized_image)
                        generated_part_samples += 1
                        per_image_face_images += 1

            needed_negative_images = int((1.0 * per_image_face_images *
                                          datasets_constants.negative_ratio) /
                                         (datasets_constants.positive_ratio +
                                          datasets_constants.part_ratio))
            needed_negative_images = max(1, needed_negative_images)

            current_negative_images = 0
            for box in detected_box:
                x_left, y_top, x_right, y_bottom, _ = box.astype(int)
                width = x_right - x_left + 1
                height = y_bottom - y_top + 1

                if ((width < minimum_face_size) or (height < minimum_face_size)
                        or (x_left < 0) or (y_top < 0)
                        or (x_right > (current_image.shape[1] - 1))
                        or (y_bottom > (current_image.shape[0] - 1))):
                    continue

                current_iou = iou(box, ground_truth_box)
                cropped_image = current_image[y_top:y_bottom +
                                              1, x_left:x_right + 1, :]
                resized_image = cv2.resize(
                    cropped_image, (target_face_size, target_face_size),
                    interpolation=cv2.INTER_LINEAR)

                if np.max(current_iou) < DatasetFactory.negative_iou() and \
                        current_negative_images < needed_negative_images:
                    file_path = os.path.join(negative_dir, f"hard-negative-{generated_negative_samples}.jpg")
                    # MEF: wrong to use linesep here
                    # negative_file.write(file_path + ' 0' + os.linesep)
                    negative_file.write(file_path + ' 0\n')
                    cv2.imwrite(file_path, resized_image)
                    generated_negative_samples += 1
                    current_negative_images += 1

            pt.update(f"{img_idx} / {total_images}")

        mef.tsprint(f"Generated {generated_positive_samples} positive samples, {generated_part_samples} partials, "
                    f"and {generated_negative_samples} negatives.")
        negative_file.close()
        part_file.close()
        positive_file.close()
        return True

    def generate_hard_samples(self, annotation_image_dir, annotation_file_name,
                              model_train_dir, network_name, minimum_face_size, target_root_dir):
        status, dataset = self._read(annotation_image_dir, annotation_file_name)
        if not status:
            return False

        test_data = InferenceBatch(dataset['images'])
        previous_network = NetworkFactory.previous_network(network_name)
        if not model_train_dir:
            model_train_dir = NetworkFactory.model_train_dir()
        mef.tsprint(f"Loading detector from {model_train_dir}...")
        face_detector = FaceDetector(previous_network, model_train_dir)
        mef.tsprint(f"Detector loaded. Detecting faces over dataset {annotation_file_name} "
                    f"of size {test_data.size}...")
        face_detector.set_min_face_size(minimum_face_size)
        face_detector.set_threshold([0.6, 0.7, 0.7])
        detected_boxes, landmarks = face_detector.detect_face(test_data, show_progress=True)
        mef.tsprint(f"Detection complete.")
        return self._generate_hard_samples(dataset, network_name, detected_boxes, minimum_face_size, target_root_dir)
