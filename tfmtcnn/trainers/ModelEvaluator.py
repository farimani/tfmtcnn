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

# import os
import numpy as np
# import cv2

from tfmtcnn.networks.FaceDetector import FaceDetector

from tfmtcnn.datasets.DatasetFactory import DatasetFactory
from tfmtcnn.datasets.InferenceBatch import InferenceBatch
import tfmtcnn.datasets.constants as datasets_constants

from tfmtcnn.utils.convert_to_square import convert_to_square
from tfmtcnn.utils.IoU import iou


class ModelEvaluator(object):
    def __init__(self):
        self._test_dataset = None
        self._face_detector = None

    def load(self, dataset_name, annotation_image_dir, annotation_file_name):
        face_dataset = DatasetFactory.face_dataset(dataset_name)
        if not face_dataset:
            return False

        if not face_dataset.read(annotation_image_dir, annotation_file_name):
            return False

        self._test_dataset = face_dataset.data()
        return True

    def create_detector(self, last_network, model_root_dir):
        self._face_detector = FaceDetector(last_network, model_root_dir)

        minimum_face_size = datasets_constants.minimum_face_size
        self._face_detector.set_min_face_size(minimum_face_size)
        return True

    def evaluate(self, print_result=False):
        if not self._test_dataset:
            return False

        if not self._face_detector:
            return False

        test_data = InferenceBatch(self._test_dataset['images'])
        detected_boxes, landmarks = self._face_detector.detect_face(test_data)

        image_file_names = self._test_dataset['images']
        ground_truth_boxes = self._test_dataset['bboxes']
        number_of_images = len(image_file_names)

        if len(detected_boxes) != number_of_images:
            return False

        assert datasets_constants.part_iou <= datasets_constants.positive_iou
        no_match_thresh = datasets_constants.part_iou

        n_pos = n_pos_sq = n_part = n_part_sq = n_labeled = n_dets = 0
        n_fp = n_fp_sq = 0  # false positives
        iou_sum = iou_sum_sq = 0.0
        for i, image_file_path in enumerate(image_file_names):
            dets, bboxes = detected_boxes[i], ground_truth_boxes[i]
            bboxes = np.array(bboxes, dtype=np.float32).reshape(-1, 4)
            n_labeled += len(bboxes)
            if dets.shape[0] == 0:
                continue

            n_dets += len(dets)
            dets_square = convert_to_square(dets)       # squared detection box
            dets_square[:, 0:4] = np.round(dets_square[:, 0:4])
            # current_image = cv2.imread(image_file_path)

            for box in bboxes:
                # x_left, y_top, x_right, y_bottom, _ = box.astype(int)
                # width = x_right - x_left + 1
                # height = y_bottom - y_top + 1

                # if( (x_left < 0) or (y_top < 0) or (x_right > (current_image.shape[1] - 1) ) or
                #         (y_bottom > (current_image.shape[0] - 1 ) ) ):
                # 	continue

                max_iou, max_iou_sq = np.max(iou(box, dets)), np.max(iou(box, dets_square))
                iou_sum += max_iou
                iou_sum_sq += max_iou_sq

                if max_iou >= datasets_constants.positive_iou:
                    n_pos += 1
                elif max_iou >= datasets_constants.part_iou:
                    n_part += 1

                if max_iou_sq >= datasets_constants.positive_iou:
                    n_pos_sq += 1
                elif max_iou_sq >= datasets_constants.part_iou:
                    n_part_sq += 1

            # MEF: Now calculate negatives. Since we don't keep track of total negatives here, we can't calculate
            #      FPR (FP/(FP + TN) where TN is true -ves). Instead we calculate and report FDR, which is the
            #      False Discovery Rate, and FDR = FP/(FP + TP). FDR is (1 - precision), where precision, or PPV,
            #      is PPV = TP/(TP + FP).
            for j, det in enumerate(dets):
                max_iou, max_iou_sq = np.max(iou(det, bboxes)), np.max(iou(dets_square[j], bboxes))
                n_fp += 1 if max_iou < no_match_thresh else 0
                n_fp_sq += 1 if max_iou_sq < no_match_thresh else 0

        if print_result:
            print(f"                       {'orig':>10s}  {'square':>10s}")
            print(f"                       {'----------':>10s}  {'---------':>10s}")
            print(f"Positive faces       - {n_pos:10d}  {n_pos_sq:10d}")
            print(f"Partial faces        - {n_part:10d}  {n_part_sq:10d}")
            print(f"Total detected faces - {(n_pos + n_part):10d}  {(n_pos_sq + n_part_sq):10d}")
            print(f"False positives count- {n_fp:10d}  {n_fp_sq:10d}")
            print(f"Ground truth faces   - {n_labeled:10d}")
            print(f"Total Detections     - {n_dets:10d}")
            print(f"Positives DR         - {n_pos / n_labeled:10.05f}  {n_pos_sq / n_labeled:10.05f}")
            print(f"Positives + Part DR  - {(n_pos+n_part)/n_labeled:10.05f}  {(n_pos_sq+n_part_sq)/n_labeled:10.05f}")
            print(f"FDR: FP/(FP+TP)      - {(n_fp)/n_dets:10.05f}  {(n_fp_sq)/n_dets:10.05f}")
            print(f"Average IOU          - {iou_sum / n_labeled:10.05f}  {iou_sum_sq / n_labeled:10.05f}")

        return True
