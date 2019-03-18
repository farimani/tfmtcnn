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
import math
import numpy as np
import cv2
import numpy.random as npr

from tfmtcnn.utils.IoU import iou

import tfmtcnn.datasets.constants as datasets_constants
from tfmtcnn.datasets.DatasetFactory import DatasetFactory
from tfmtcnn.utils.convert_to_square import convert_to_square

import mef


class SimpleFaceDataset(object):
    __positive_ratio = datasets_constants.positive_ratio
    __part_ratio = datasets_constants.part_ratio
    __negative_ratio = datasets_constants.negative_ratio

    def __init__(self, name='SimpleFaceDataset'):
        self._name = name

    @classmethod
    def positive_file_name(cls, target_root_dir):
        positive_file_name = os.path.join(target_root_dir, 'positive.txt')
        return positive_file_name

    @classmethod
    def part_file_name(cls, target_root_dir):
        part_file_name = os.path.join(target_root_dir, 'part.txt')
        return part_file_name

    @classmethod
    def negative_file_name(cls, target_root_dir):
        negative_file_name = os.path.join(target_root_dir, 'negative.txt')
        return negative_file_name

    # def is_valid(self):
    #     return self._is_valid
    #
    # def data(self):
    #     return self._data

    @staticmethod
    def _read(annotation_image_dir, annotation_file_name, face_dataset_name='WIDERFaceDataset'):
        dataset = None
        status = False
        face_dataset = DatasetFactory.face_dataset(face_dataset_name)
        if face_dataset.read(annotation_image_dir, annotation_file_name):
            dataset = face_dataset.data()
            status = True

        return status, dataset

    @staticmethod
    def _generated_samples(target_root_dir):
        # MEF: Replace these with a more robust line counter. readlines() generate extra lines in this code on
        #      windows

        # positive_file = open(SimpleFaceDataset.positive_file_name(target_root_dir), 'a+')
        # generated_positive_samples = len(positive_file.readlines())
        # part_file = open(SimpleFaceDataset.part_file_name(target_root_dir), 'a+')
        # generated_part_samples = len(part_file.readlines())
        # negative_file = open(SimpleFaceDataset.negative_file_name(target_root_dir), 'a+')
        # generated_negative_samples = len(negative_file.readlines())
        # negative_file.close()
        # part_file.close()
        # positive_file.close()

        try:
            generated_positive_samples = mef.get_line_count(SimpleFaceDataset.positive_file_name(target_root_dir))
        except Exception:
            generated_positive_samples = 0

        try:
            generated_part_samples = mef.get_line_count(SimpleFaceDataset.part_file_name(target_root_dir))
        except Exception:
            generated_part_samples = 0

        try:
            generated_negative_samples = mef.get_line_count(SimpleFaceDataset.negative_file_name(target_root_dir))
        except Exception:
            generated_negative_samples = 0

        return generated_positive_samples, generated_part_samples, generated_negative_samples

    def generate_simple_samples(self, annotation_image_dir, annotation_file_name, base_number_of_images,
                                target_face_size, target_root_dir):
        status, dataset = self._read(annotation_image_dir, annotation_file_name)
        if not status:
            return False

        image_file_names = dataset['images']
        ground_truth_boxes = dataset['bboxes']
        number_of_faces = dataset['number_of_faces']

        positive_dir = os.path.join(target_root_dir, 'positive')
        part_dir = os.path.join(target_root_dir, 'part')
        negative_dir = os.path.join(target_root_dir, 'negative')

        mef.create_dir_if_necessary(positive_dir, raise_on_error=True)
        mef.create_dir_if_necessary(part_dir, raise_on_error=True)
        mef.create_dir_if_necessary(negative_dir, raise_on_error=True)

        generated_positive_samples, generated_part_samples, \
            generated_negative_samples = self._generated_samples(target_root_dir)

        positive_file = open(SimpleFaceDataset.positive_file_name(target_root_dir), 'a+')
        part_file = open(SimpleFaceDataset.part_file_name(target_root_dir), 'a+')
        negative_file = open(SimpleFaceDataset.negative_file_name(target_root_dir), 'a+')

        negative_samples_per_image_ratio = (SimpleFaceDataset.__negative_ratio - 1)
        needed_negative_samples = base_number_of_images - (generated_negative_samples /
                                                           (1.0 * SimpleFaceDataset.__negative_ratio))
        needed_base_negative_samples = (1.0 * needed_negative_samples) / number_of_faces
        needed_negative_samples_per_image = int(1.0 * negative_samples_per_image_ratio *
                                                needed_base_negative_samples *
                                                (1.0 * number_of_faces / len(image_file_names)))
        needed_negative_samples_per_image = max(0, needed_negative_samples_per_image)

        needed_negative_samples_per_bounding_box = np.ceil(1.0 * (SimpleFaceDataset.__negative_ratio -
                                                                  negative_samples_per_image_ratio) *
                                                           needed_base_negative_samples)
        needed_negative_samples_per_bounding_box = max(0, needed_negative_samples_per_bounding_box)
        needed_positive_samples = (base_number_of_images * SimpleFaceDataset.__positive_ratio) - \
            generated_positive_samples
        needed_positive_samples_per_bounding_box = np.ceil(1.0 * needed_positive_samples / number_of_faces)
        needed_positive_samples_per_bounding_box = max(0, needed_positive_samples_per_bounding_box)
        needed_part_samples = (base_number_of_images * SimpleFaceDataset.__part_ratio) - generated_part_samples
        needed_part_samples_per_bounding_box = np.ceil(1.0 * needed_part_samples / number_of_faces)
        needed_part_samples_per_bounding_box = max(0, needed_part_samples_per_bounding_box)

        # MEF: 20 is a good tradeoff. We get the speed with very few samples sacrificed for positives (~1/1000 faces)
        base_number_of_attempts = 20  # 5000
        current_image_number = 0

        # MEF: if you want to resume from an interrupted loop, set this appropriately. Make sure to interrupt at the
        # outer loop to keep things in order!
        skip_count = 0
        pt = mef.ProgressText(len(image_file_names) - skip_count)

        # MEF: Use the following metrics to get a reasonable number for base_number_of_attempts.
        skipped_negatives = skipped_partials = skipped_positives = 0
        zero_negatives = zero_partials = zero_positives = 0

        # for image_file_path, ground_truth_box in zip(image_file_names, ground_truth_boxes):
        for i in range(skip_count, len(image_file_names)):
            image_file_path, ground_truth_box = image_file_names[i], ground_truth_boxes[i]
            bounding_boxes = np.array(ground_truth_box, dtype=np.float32).reshape(-1, 4)
            current_image = cv2.imread(image_file_path)
            input_image_height, input_image_width, input_image_channels = current_image.shape

            needed_negative_samples = needed_negative_samples_per_image
            negative_images = 0
            maximum_attempts = base_number_of_attempts * needed_negative_samples
            number_of_attempts = 0
            while negative_images < needed_negative_samples and number_of_attempts < maximum_attempts:
                number_of_attempts += 1
                crop_size = npr.randint(target_face_size, min(input_image_width, input_image_height) / 2)
                nx = npr.randint(0, (input_image_width - crop_size))
                ny = npr.randint(0, (input_image_height - crop_size))
                crop_box = np.array([nx, ny, nx + crop_size, ny + crop_size])
                current_iou = iou(crop_box, bounding_boxes)

                if np.max(current_iou) < DatasetFactory.negative_iou():
                    cropped_image = current_image[ny:ny + crop_size, nx:nx + crop_size, :]
                    resized_image = cv2.resize(cropped_image, (target_face_size, target_face_size),
                                               interpolation=cv2.INTER_LINEAR)
                    file_path = os.path.join(negative_dir, "simple-negative-%s.jpg" % generated_negative_samples)
                    # negative_file.write(file_path + ' 0' + os.linesep)    # MEF: wrong to use linesep here
                    negative_file.write(file_path + ' 0\n')
                    cv2.imwrite(file_path, resized_image)
                    generated_negative_samples += 1
                    negative_images += 1

            if negative_images == 0:
                zero_negatives += 1
            elif negative_images < needed_negative_samples:
                skipped_negatives += 1

            needed_negative_samples = needed_negative_samples_per_bounding_box
            needed_positive_samples = needed_positive_samples_per_bounding_box
            needed_part_samples = needed_part_samples_per_bounding_box
            for bounding_box in bounding_boxes:
                x1, y1, x2, y2 = bounding_box
                bounding_box_width = x2 - x1 + 1
                bounding_box_height = y2 - y1 + 1

                if x1 < 0 or y1 < 0:
                    continue

                negative_images = 0
                maximum_attempts = base_number_of_attempts * needed_negative_samples
                number_of_attempts = 0
                while negative_images < needed_negative_samples and number_of_attempts < maximum_attempts:
                    number_of_attempts += 1
                    crop_size = npr.randint(target_face_size, min(input_image_width, input_image_height) / 2)
                    delta_x = npr.randint(max(-1 * crop_size, -1 * x1), bounding_box_width)
                    delta_y = npr.randint(max(-1 * crop_size, -1 * y1), bounding_box_height)

                    # delta_x = npr.randint(-1 * crop_size, +1 * crop_size + 1) * 0.2
                    # delta_y = npr.randint(-1 * crop_size, +1 * crop_size + 1) * 0.2

                    nx1 = int(max(0, x1 + delta_x))
                    ny1 = int(max(0, y1 + delta_y))
                    if (nx1 + crop_size) > input_image_width or (ny1 + crop_size) > input_image_height:
                        continue

                    crop_box = np.array([nx1, ny1, nx1 + crop_size, ny1 + crop_size])
                    current_iou = iou(crop_box, bounding_boxes)

                    if np.max(current_iou) < DatasetFactory.negative_iou():
                        cropped_image = current_image[ny1:ny1 + crop_size, nx1:nx1 + crop_size, :]
                        resized_image = cv2.resize(cropped_image, (target_face_size, target_face_size),
                                                   interpolation=cv2.INTER_LINEAR)
                        file_path = os.path.join(negative_dir, "simple-negative-%s.jpg" % generated_negative_samples)
                        # negative_file.write(file_path + ' 0' + os.linesep)   # MEF: wrong to use linesep here
                        negative_file.write(file_path + ' 0\n')
                        cv2.imwrite(file_path, resized_image)
                        generated_negative_samples += 1
                        negative_images += 1

                if negative_images == 0:
                    zero_negatives += 1
                elif negative_images < needed_negative_samples:
                    skipped_negatives += 1

                positive_images = part_images = 0
                maximum_attempts = base_number_of_attempts * (needed_positive_samples + needed_part_samples)
                number_of_attempts = 0

                pt2 = mef.ProgressText(needed_positive_samples + needed_part_samples, newline_when_done=False)
                pt.show(clean=True)

                def update_progress(pt2_updated):
                    """ Call when short circuting the loop"""
                    pt.update_current_time(show=False)

                    if number_of_attempts < maximum_attempts * 0.5:     # don't be too noisy with the second progress
                        return

                    msg = pt.get_output_string() + f" - Positives: {positive_images}/{needed_positive_samples}, " \
                        f"Partials: {part_images}/{needed_part_samples} {pt.percent_done():4.1f}%"
                    if pt2_updated:
                        pt2.update(msg)
                    else:
                        pt2.update_current_time(msg)

                while number_of_attempts < maximum_attempts and \
                        (positive_images < needed_positive_samples or part_images < needed_part_samples):
                    number_of_attempts += 1
                    # MEF: The following formula is too simple. In practice, you get many situations where
                    #      the overlap of a square crop box with a rectangular truth box will never meet a given
                    #      criteria of say, 0.65 IOU. Specifically, the formulas are as follows:
                    #
                    #                   iou <= 1 / (2 * sqrt(aspect_ratio) - 1)
                    #               ==> aspect_ratio <= (((1/iou) + 1) / 2)^2
                    #
                    #                   where aspect_ratio is the >= 1 aspect ratio.
                    #
                    #      So, for a 0.65 iou, the max aspect ratio is ~1.61.
                    #
                    #      We modify the algorithm to pick a rectangular crop box, do the IOU calc, and once
                    #      we find a good match, we convert the crop box to square.
                    #      Even then, we will have cases that making the box square will always put the crop box
                    #      extending beyond image boundaries.

                    # crop_size = npr.randint(int(min(bounding_box_width, bounding_box_height) * 0.8),
                    #                         np.ceil(1.25 * max(bounding_box_width, bounding_box_height)))
                    crop_w = npr.randint(int(bounding_box_width * 0.8), np.ceil(bounding_box_width * 1.25))
                    crop_h = npr.randint(int(bounding_box_height * 0.8), np.ceil(bounding_box_height * 1.25))

                    delta_x = npr.randint(-1.0 * bounding_box_width, +1.0 * bounding_box_width + 1) * 0.2
                    delta_y = npr.randint(-1.0 * bounding_box_height, +1.0 * bounding_box_height + 1) * 0.2

                    nx1 = int(max((x1 + bounding_box_width / 2.0 + delta_x - crop_w / 2.0), 0))
                    ny1 = int(max((y1 + bounding_box_height / 2.0 + delta_y - crop_h / 2.0), 0))
                    nx2, ny2 = nx1 + crop_w, ny1 + crop_h

                    if nx2 > input_image_width or ny2 > input_image_height:
                        update_progress(False)
                        continue

                    crop_box = np.array([nx1, ny1, nx2, ny2])
                    nx1, ny1, nx2, ny2 = convert_to_square(crop_box.reshape(1, -1))[0]  # now convert to square

                    if nx1 < 0 or ny1 < 0 or nx2 > input_image_width or ny2 > input_image_height:   # out of bounds?
                        update_progress(False)
                        continue

                    iou_amount = iou(crop_box, bounding_box.reshape(1, -1))
                    updated = False

                    if (iou_amount >= DatasetFactory.part_iou() and part_images < needed_part_samples) or \
                            (iou_amount >= DatasetFactory.positive_iou() and positive_images < needed_positive_samples):
                        crop_size = nx2 - nx1 + 1
                        offset_x1, offset_y1 = (x1 - nx1) / crop_size, (y1 - ny1) / crop_size
                        offset_x2, offset_y2 = (x2 - nx2) / crop_size, (y2 - ny2) / crop_size
                        updated = True
                        cropped_image = current_image[ny1:ny2, nx1:nx2, :]
                        resized_image = cv2.resize(cropped_image, (target_face_size, target_face_size),
                                                   interpolation=cv2.INTER_LINEAR)
                        if iou_amount >= DatasetFactory.positive_iou() and positive_images < needed_positive_samples:
                            file_path = os.path.join(positive_dir, f"simple-positive-{generated_positive_samples}.jpg")
                            # MEF: wrong to use linesep here
                            # positive_file.write(file_path + ' 1 %.2f %.2f %.2f %.2f' %
                            #                     (offset_x1, offset_y1, offset_x2, offset_y2) + os.linesep)
                            positive_file.write(f"{file_path} 1 {offset_x1:.2f} {offset_y1:.2f} "
                                                f"{offset_x2:.2f} {offset_y2:.2f}\n")
                            cv2.imwrite(file_path, resized_image)
                            generated_positive_samples += 1
                            positive_images += 1
                        else:
                            assert part_images < needed_part_samples and iou_amount >= DatasetFactory.part_iou()
                            file_path = os.path.join(part_dir, f"simple-part-{generated_part_samples}.jpg")
                            # MEF: wrong to use linesep here
                            # part_file.write(file_path + ' -1 %.2f %.2f %.2f %.2f' %
                            #                 (offset_x1, offset_y1, offset_x2, offset_y2) + os.linesep)
                            part_file.write(f"{file_path} -1 {offset_x1:.2f} {offset_y1:.2f} "
                                            f"{offset_x2:.2f} {offset_y2:.2f}\n")
                            cv2.imwrite(file_path, resized_image)
                            generated_part_samples += 1
                            part_images += 1

                    update_progress(updated)

                if part_images == 0:
                    zero_partials += 1
                elif part_images < needed_part_samples:
                    skipped_partials += 1

                if positive_images == 0:
                    zero_positives += 1
                elif positive_images < needed_positive_samples:
                    skipped_positives += 1

                pt.show(clean=True)       # show main progress text

            current_image_number += 1
            if current_image_number % 1000 == 0:
                print(f"\r{current_image_number}/{len(image_file_names)} done - "
                      f"positive - {generated_positive_samples},  part - {generated_part_samples}, "
                      f"negative - {generated_negative_samples}")
                # MEF: Use the following metrics to get a reasonable number for base_number_of_attempts.
                # print(f"zero -ves: {zero_negatives}, skipped -ves:{skipped_negatives}")
                # print(f"zero +ves: {zero_positives}, skipped +ves:{skipped_positives}")
                # print(f"zero parts: {zero_partials}, skipped parts:{skipped_partials}")
            pt.update()

        negative_file.close()
        part_file.close()
        positive_file.close()
        return True
