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
import time
import math
import cv2
import numpy as np

from tfmtcnn.utils.nms import py_nms
from tfmtcnn.utils.convert_to_square import convert_to_square
from tfmtcnn.utils.IoU import iou

import tfmtcnn.datasets.constants as datasets_constants
from tfmtcnn.datasets.DatasetFactory import DatasetFactory
from tfmtcnn.datasets.InferenceBatch import InferenceBatch

from tfmtcnn.networks.NetworkFactory import NetworkFactory

import mef

class FaceDetector(object):
    def __init__(self, last_network='ONet', model_root_dir=None):
        if not model_root_dir:
            self._model_root_dir = NetworkFactory.model_deploy_dir()
        else:
            self._model_root_dir = model_root_dir

        self._min_face_size = datasets_constants.minimum_face_size
        self._threshold = [0.9, 0.6, 0.7]
        # self._scale_factor = 0.79
        self._scale_factor = 1.0 / (math.sqrt(2.0))

        status_ok = True
        if last_network in ['PNet', 'RNet', 'ONet']:
            self._pnet = NetworkFactory.network('PNet')
            pnet_model_path = os.path.join(self._model_root_dir,
                                           self._pnet.network_name())
            status_ok = self._pnet.setup_inference_network(
                pnet_model_path) and status_ok
        else:
            self._pnet = None

        if last_network in ['RNet', 'ONet']:
            self._rnet = NetworkFactory.network('RNet')
            rnet_model_path = os.path.join(self._model_root_dir,
                                           self._rnet.network_name())
            status_ok = self._rnet.setup_inference_network(
                rnet_model_path) and status_ok
        else:
            self._rnet = None

        if last_network in ['ONet']:
            self._onet = NetworkFactory.network('ONet')
            onet_model_path = os.path.join(self._model_root_dir,
                                           self._onet.network_name())
            status_ok = self._onet.setup_inference_network(
                onet_model_path) and status_ok
        else:
            self._onet = None

        if not status_ok:
            raise SystemExit

    def set_threshold(self, threshold):
        self._threshold = threshold

    def set_min_face_size(self, min_face_size):
        self._min_face_size = min_face_size

    @staticmethod
    def _generate_bbox(cls_map, reg, scale, threshold):
        stride = 2
        # stride = 4
        cellsize = 12
        # cellsize = 25
        t_index = np.where(cls_map > threshold)

        # find nothing
        if t_index[0].size == 0:
            return np.array([])
        # offset
        dx1, dy1, dx2, dy2 = [reg[t_index[0], t_index[1], i] for i in range(4)]

        reg = np.array([dx1, dy1, dx2, dy2])
        score = cls_map[t_index[0], t_index[1]]
        boundingbox = np.vstack([
            np.round((stride * t_index[1]) / scale),
            np.round((stride * t_index[0]) / scale),
            np.round((stride * t_index[1] + cellsize) / scale),
            np.round((stride * t_index[0] + cellsize) / scale), score, reg
        ])

        return boundingbox.T

    @staticmethod
    def _processed_image(image, scale):
        height, width, channels = image.shape
        new_height = int(height * scale)
        new_width = int(width * scale)
        new_shape = (new_width, new_height)
        resized_image = cv2.resize(image, new_shape, interpolation=cv2.INTER_LINEAR)
        resized_image = (resized_image - 127.5) / 128
        return resized_image

    @staticmethod
    def _pad(bboxes, w, h):
        tmpw, tmph = (bboxes[:, 2] - bboxes[:, 0] + 1), (bboxes[:, 3] - bboxes[:, 1] + 1)
        num_box = bboxes.shape[0]

        dx, dy = np.zeros((num_box, )), np.zeros((num_box, ))
        edx, edy = tmpw.copy() - 1, tmph.copy() - 1

        x, y, ex, ey = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

        tmp_index = np.where(ex > w - 1)
        edx[tmp_index] = tmpw[tmp_index] + w - 2 - ex[tmp_index]
        ex[tmp_index] = w - 1

        tmp_index = np.where(ey > h - 1)
        edy[tmp_index] = tmph[tmp_index] + h - 2 - ey[tmp_index]
        ey[tmp_index] = h - 1

        tmp_index = np.where(x < 0)
        dx[tmp_index] = 0 - x[tmp_index]
        x[tmp_index] = 0

        tmp_index = np.where(y < 0)
        dy[tmp_index] = 0 - y[tmp_index]
        y[tmp_index] = 0

        return_list = [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
        return_list = [item.astype(np.int32) for item in return_list]

        return return_list

    @staticmethod
    def _calibrate_box(bbox, reg):
        bbox_c = bbox.copy()
        w = bbox[:, 2] - bbox[:, 0] + 1
        w = np.expand_dims(w, 1)
        h = bbox[:, 3] - bbox[:, 1] + 1
        h = np.expand_dims(h, 1)
        reg_m = np.hstack([w, h, w, h])
        aug = reg_m * reg
        bbox_c[:, 0:4] = bbox_c[:, 0:4] + aug
        return bbox_c

    def _propose_faces(self, image):
        # h, w, c = image.shape
        net_size = self._pnet.network_size()

        current_scale = float(net_size) / self._min_face_size
        resized_image = self._processed_image(image, current_scale)
        current_height, current_width, _ = resized_image.shape

        all_boxes = list()
        while min(current_height, current_width) >= net_size:
            cls_cls_map, reg = self._pnet.detect(resized_image)
            boxes = self._generate_bbox(cls_cls_map[:, :, 1], reg, current_scale, self._threshold[0])

            current_scale *= self._scale_factor
            resized_image = self._processed_image(image, current_scale)
            current_height, current_width, _ = resized_image.shape

            if boxes.size == 0:
                continue
            keep = py_nms(boxes[:, :5], 0.5, 'Union')
            boxes = boxes[keep]
            all_boxes.append(boxes)

        if len(all_boxes) == 0:
            return None, None, None

        all_boxes = np.vstack(all_boxes)

        # merge the detection from first stage
        keep = py_nms(all_boxes[:, 0:5], 0.7, 'Union')
        all_boxes = all_boxes[keep]
        boxes = all_boxes[:, :5]

        bbw = all_boxes[:, 2] - all_boxes[:, 0] + 1
        bbh = all_boxes[:, 3] - all_boxes[:, 1] + 1

        # refine the boxes
        boxes_c = np.vstack([
            all_boxes[:, 0] + all_boxes[:, 5] * bbw,
            all_boxes[:, 1] + all_boxes[:, 6] * bbh,
            all_boxes[:, 2] + all_boxes[:, 7] * bbw,
            all_boxes[:, 3] + all_boxes[:, 8] * bbh, all_boxes[:, 4]
        ])
        boxes_c = boxes_c.T

        return boxes, boxes_c, None

    def _refine_faces(self, im, dets):
        h, w, c = im.shape
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])

        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self._pad(dets, w, h)
        num_boxes = dets.shape[0]
        cropped_ims = np.zeros((num_boxes, 24, 24, 3), dtype=np.float32)
        for i in range(num_boxes):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            cropped_ims[i, :, :, :] = (cv2.resize(tmp, (24, 24)) - 127.5) / 128
        cls_scores, reg, _ = self._rnet.detect(cropped_ims)
        cls_scores = cls_scores[:, 1]
        keep_inds = np.where(cls_scores > self._threshold[1])[0]
        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            boxes[:, 4] = cls_scores[keep_inds]
            reg = reg[keep_inds]
            # landmark = landmark[keep_inds]
        else:
            return None, None, None

        keep = py_nms(boxes, 0.7, 'Union')
        boxes = boxes[keep]
        boxes_c = self._calibrate_box(boxes, reg[keep])
        return boxes, boxes_c, None

    def _outpute_faces(self, im, dets):
        h, w, c = im.shape
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self._pad(dets, w, h)
        num_boxes = dets.shape[0]
        cropped_ims = np.zeros((num_boxes, 48, 48, 3), dtype=np.float32)
        for i in range(num_boxes):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] +
                1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            cropped_ims[i, :, :, :] = (cv2.resize(tmp, (48, 48)) - 127.5) / 128

        cls_scores, reg, landmark = self._onet.detect(cropped_ims)
        cls_scores = cls_scores[:, 1]
        keep_inds = np.where(cls_scores > self._threshold[2])[0]
        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            boxes[:, 4] = cls_scores[keep_inds]
            reg = reg[keep_inds]
            landmark = landmark[keep_inds]
        else:
            return None, None, None

        w = boxes[:, 2] - boxes[:, 0] + 1
        h = boxes[:, 3] - boxes[:, 1] + 1

        landmark[:, 0::2] = (np.tile(w,
                                     (5, 1)) * landmark[:, 0::2].T + np.tile(
                                         boxes[:, 0], (5, 1)) - 1).T
        landmark[:, 1::2] = (np.tile(h,
                                     (5, 1)) * landmark[:, 1::2].T + np.tile(
                                         boxes[:, 1], (5, 1)) - 1).T
        boxes_c = self._calibrate_box(boxes, reg)

        boxes = boxes[py_nms(boxes, 0.7, "Minimum")]
        keep = py_nms(boxes_c, 0.7, "Minimum")
        boxes_c = boxes_c[keep]
        landmark = landmark[keep]
        return boxes, boxes_c, landmark

    # noinspection PyUnusedLocal
    def detect(self, image):
        boxes_c = landmark = None
        start_time = time.time()
        if self._pnet:
            boxes, boxes_c, _ = self._propose_faces(image)
            if boxes_c is None:
                return np.array([]), np.array([])
            pnet_time = time.time() - start_time
        else:
            return boxes_c, landmark

        start_time = time.time()
        if self._rnet:
            boxes, boxes_c, _ = self._refine_faces(image, boxes_c)
            if boxes_c is None:
                return np.array([]), np.array([])
            rnet_time = time.time() - start_time
        else:
            return boxes_c, landmark

        start_time = time.time()
        if self._onet:
            boxes, boxes_c, landmark = self._outpute_faces(image, boxes_c)
            if boxes_c is None:
                return np.array([]), np.array([])
            onet_time = time.time() - start_time

        return boxes_c, landmark

    def detect_face(self, data_batch, show_progress=False):
        all_boxes_c = []
        all_landmarks = []

        pt = mef.ProgressText(data_batch.size) if show_progress else None

        for image in data_batch:
            boxes_c, landmarks = self.detect(image)
            all_boxes_c.append(boxes_c)
            all_landmarks.append(landmarks)

            if pt is not None:
                pt.update()

        return all_boxes_c, all_landmarks

    def evaluate_dataset(self, test_dataset, print_result=False):

        test_data = InferenceBatch(test_dataset['images'])
        detected_boxes, landmarks = self.detect_face(test_data)

        image_file_names = test_dataset['images']
        ground_truth_boxes = test_dataset['bboxes']
        number_of_images = len(image_file_names)

        if len(detected_boxes) != number_of_images:
            return False

        number_of_positive_faces = 0
        number_of_part_faces = 0
        number_of_ground_truth_faces = 0
        accuracy = 0.0
        for image_file_path, detected_box, ground_truth_box in zip(
                image_file_names, detected_boxes, ground_truth_boxes):
            ground_truth_box = np.array(
                ground_truth_box, dtype=np.float32).reshape(-1, 4)
            number_of_ground_truth_faces = number_of_ground_truth_faces + len(
                ground_truth_box)
            if detected_box.shape[0] == 0:
                continue

            detected_box = convert_to_square(detected_box)
            detected_box[:, 0:4] = np.round(detected_box[:, 0:4])

            current_image = cv2.imread(image_file_path)

            for box in detected_box:
                x_left, y_top, x_right, y_bottom, _ = box.astype(int)
                # width = x_right - x_left + 1
                # height = y_bottom - y_top + 1

                if ((x_left < 0) or (y_top < 0)
                        or (x_right > (current_image.shape[1] - 1))
                        or (y_bottom > (current_image.shape[0] - 1))):
                    continue

                current_iou = iou(box, ground_truth_box)
                maximum_iou = np.max(current_iou)

                accuracy = accuracy + maximum_iou
                if maximum_iou > datasets_constants.positive_iou:
                    number_of_positive_faces = number_of_positive_faces + 1
                elif maximum_iou > datasets_constants.part_iou:
                    number_of_part_faces = number_of_part_faces + 1

        if print_result:
            print('Positive faces       - ', number_of_positive_faces)
            print('Partial faces        - ', number_of_part_faces)
            print('Total detected faces - ', (number_of_positive_faces + number_of_part_faces))
            print('Ground truth faces   - ', number_of_ground_truth_faces)
            print('Positive accuracy    - ', number_of_positive_faces / number_of_ground_truth_faces)
            print('Detection accuracy   - ',
                  (number_of_positive_faces + number_of_part_faces) / number_of_ground_truth_faces)
            print('Accuracy             - ', accuracy / number_of_ground_truth_faces)

        return True

    def evaluate(self, dataset_name, annotation_image_dir, annotation_file_name, print_result=False):
        # test_dataset = None
        face_dataset = DatasetFactory.face_dataset(dataset_name)
        if face_dataset.read(annotation_image_dir, annotation_file_name):
            test_dataset = face_dataset.data()
        else:
            return False

        return self.evaluate_dataset(test_dataset, print_result)
