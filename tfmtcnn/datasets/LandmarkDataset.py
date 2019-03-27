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
from os.path import join
import random
import cv2
import numpy as np
import numpy.random as npr

from tfmtcnn.utils.BBox import BBox
from tfmtcnn.utils.IoU import iou

import datasets.constants as datasets_constants
from tfmtcnn.datasets.DatasetFactory import DatasetFactory

from tfmtcnn.datasets.Landmark import rotate
from tfmtcnn.datasets.Landmark import flip
from tfmtcnn.utils.convert_to_square import convert_to_square

# from tfmtcnn.datasets.Landmark import random_shift
# from tfmtcnn.datasets.Landmark import random_shift_with_argument

import mef


class LandmarkDataset(object):
    def __init__(self, name='Landmark'):
        self._name = name
        self._is_valid = False
        self._data = {}

        # MEF: Avoid checking on directory existence for every image write by keeping track...
        self._dirs_set = set()
        self._clear()

    def _clear(self):
        self._is_valid = False
        self._data = {}
        self._dirs_set = set()

    @classmethod
    def landmark_file_name(cls, target_root_dir):
        landmark_file_name = os.path.join(target_root_dir, 'landmark.txt')
        return landmark_file_name

    def is_valid(self):
        return self._is_valid

    def data(self):
        return self._data

    def _read(self, landmark_image_dir, landmark_file_name):
        self._clear()

        # landmark_dataset = DatasetFactory.landmark_dataset('LFWLandmark')
        landmark_dataset = DatasetFactory.landmark_dataset('CelebADataset')
        if landmark_dataset.read(landmark_image_dir, landmark_file_name):
            self._is_valid = True
            self._data = landmark_dataset.data()

        return self._is_valid

    @staticmethod
    def _can_generate_sample():
        return random.choice([0, 1, 2, 3]) > 1

    def _make_image_fname(self, root_dir, target_size, counter, prefix, ext):
        """ Avoids super large directories by putting a max number of files per directory.
            It creates the directory if needed.
        """
        max_files_per_dir = 3000
        dirpath = os.path.join(root_dir, f"{target_size}x{target_size}", f"dir{counter // max_files_per_dir}")

        if dirpath not in self._dirs_set:
            mef.create_dir_if_necessary(dirpath)
            self._dirs_set.add(dirpath)

        fname = os.path.join(dirpath, f"{prefix}-{counter}.{ext}")
        return fname

    def _resize_and_save(self, root_dir, image, target_size, counter, prefix):
        fname = self._make_image_fname(root_dir, target_size, counter, prefix, "jpg")
        resized = mef.resize_image(image, target_size, target_size)
        cv2.imwrite(fname, resized)

        # other_sizes = [x for x in [12, 24, 48] if x != target_size]
        # for size in other_sizes:
        #     fn = self._make_image_fname(root_dir, size, counter, prefix, "jpg")
        #     cv2.imwrite(fn, mef.resize_image(image, size, size))

        return fname

    def generate(self, landmark_image_dir, landmark_file_name, base_number_of_images,
                 target_face_size, target_root_dir):
        if not self._read(landmark_image_dir, landmark_file_name):
            return False

        # MEF: CelebA has one markup per image.
        #      Also, note that the bounding boxes are generated such that right and bottom are passed
        #      the width and height. i.e. right = left + width.
        #
        image_file_names = self._data['images']
        gt_bboxes = self._data['bboxes']
        gt_landmarks = self._data['landmarks']

        assert len(image_file_names) == len(gt_bboxes) == len(gt_landmarks), "Inconsistent landmark data."

        landmark_dir = os.path.join(target_root_dir, 'landmark')
        mef.create_dir_if_necessary(landmark_dir, raise_on_error=True)

        landmark_file = open(LandmarkDataset.landmark_file_name(target_root_dir), 'w')

        processed_input_images = generated_landmark_images = 0
        total_number_of_input_images = len(image_file_names)
        needed_landmark_samples = int((base_number_of_images * datasets_constants.landmark_parts) /
                                      total_number_of_input_images)
        needed_landmark_samples = max(1, needed_landmark_samples)
        base_number_of_attempts = 500
        maximum_attempts = base_number_of_attempts * needed_landmark_samples

        pt = mef.ProgressText(len(image_file_names))

        for i, image_path in enumerate(image_file_names):
            bbox = gt_bboxes[i]                         # MEF: one markup per image
            landmarks = gt_landmarks[i]

            assert len(landmarks) == 5, "Expect 5 landmark points."

            image_path = image_path.replace("\\", '/')
            image = cv2.imread(image_path)
            if image is None:
                print(f"WARNING: Could not parse image {image_path}. Skipping...")
                pt.update_current_time()
                continue

            image_height, image_width, _ = image.shape

            # MEF: bbox right/bottom are inclusive
            if bbox.left < 0 or bbox.top < 0 or bbox.right >= image_width or bbox.bottom >= image_height:
                pt.update_current_time()
                continue

            gt_box = np.array([bbox.left, bbox.top, bbox.right, bbox.bottom])
            gt_width, gt_height = bbox.w, bbox.h

            current_face_images, current_face_landmarks = [], []
            f_face = image[bbox.top:bbox.bottom+1, bbox.left:bbox.right+1]
            landmarks_norm = np.zeros((5, 2))       # MEF: Normalized landmarks

            for index, one in enumerate(landmarks):
                landmarks_norm[index] = ((one[0] - gt_box[0]) / gt_width, (one[1] - gt_box[1]) / gt_height)

            current_face_images.append(f_face)
            current_face_landmarks.append(landmarks_norm.reshape(10))

            current_landmark_samples = number_of_attempts = 0
            while current_landmark_samples < needed_landmark_samples and number_of_attempts < maximum_attempts:
                number_of_attempts += 1
                # MEF: See generate_simple_samples() for comments on why we change the below.
                #
                # bounding_box_size = npr.randint(int(min(ground_truth_width, ground_truth_height) * 0.8),
                #                                 np.ceil(1.25 * max(ground_truth_width, ground_truth_height)))
                bb_w = npr.randint(int(gt_width * 0.8), np.ceil(gt_width * 1.25))
                bb_h = npr.randint(int(gt_height * 0.8), np.ceil(gt_height * 1.25))

                delta_x = npr.randint(-gt_width, gt_width) * 0.2
                delta_y = npr.randint(-gt_height, gt_height) * 0.2
                nx1 = int(max(gt_box[0] + gt_width / 2 - bb_w / 2 + delta_x, 0))
                ny1 = int(max(gt_box[1] + gt_height / 2 - bb_h / 2 + delta_y, 0))
                nx2, ny2 = nx1 + bb_w - 1, ny1 + bb_h - 1

                if nx2 >= image_width or ny2 >= image_height:
                    pt.update_current_time()
                    continue

                crop_box = np.array([nx1, ny1, nx2, ny2])
                nx1, ny1, nx2, ny2 = convert_to_square(crop_box.reshape(1, -1))[0]  # now convert to square

                if nx1 < 0 or ny1 < 0 or nx2 >= image_width or ny2 >= image_height:  # out of bounds?
                    pt.update_current_time()
                    continue

                # MEF: Note that celeba is only used for landmark localization, not class training. So we only
                #      look at the positive crops here...
                #
                current_iou = iou(crop_box, np.expand_dims(gt_box, 0))
                if current_iou < DatasetFactory.positive_iou():
                    pt.update_current_time()
                    continue

                crop_size = nx2 - nx1 + 1
                cropped_im = image[ny1:ny2+1, nx1:nx2+1, :]

                if len(landmarks) != 5:
                    print("here")

                # assert len(landmarks) == 5

                for index, one in enumerate(landmarks):
                    landmarks_norm[index] = ((one[0] - nx1) / crop_size, (one[1] - ny1) / crop_size)

                current_face_images.append(cropped_im)
                current_face_landmarks.append(landmarks_norm.reshape(10))
                landmark_ = current_face_landmarks[-1].reshape(-1, 2)
                bounding_box = BBox([nx1, ny1, nx2, ny2])

                # mirror
                if self._can_generate_sample():
                    face_flipped, landmark_flipped = flip(cropped_im, landmark_)
                    # c*h*w
                    current_face_images.append(face_flipped)
                    current_face_landmarks.append(landmark_flipped.reshape(10))

                # rotate
                if self._can_generate_sample():
                    face_rotated, landmark_rotated = rotate(image, bounding_box,
                                                            bounding_box.reproject_landmark(landmark_), 5)
                    # landmark_offset
                    landmark_rotated = bounding_box.project_landmark(landmark_rotated)
                    current_face_images.append(face_rotated)
                    current_face_landmarks.append(landmark_rotated.reshape(10))

                    # flip
                    face_flipped, landmark_flipped = flip(face_rotated, landmark_rotated)
                    current_face_images.append(face_flipped)
                    current_face_landmarks.append(landmark_flipped.reshape(10))

                # inverse clockwise rotation
                if self._can_generate_sample():
                    face_rotated, landmark_rotated = rotate(image, bounding_box,
                                                            bounding_box.reproject_landmark(landmark_), -5)
                    landmark_rotated = bounding_box.project_landmark(landmark_rotated)
                    current_face_images.append(face_rotated)
                    current_face_landmarks.append(landmark_rotated.reshape(10))

                    face_flipped, landmark_flipped = flip(face_rotated, landmark_rotated)
                    current_face_images.append(face_flipped)
                    current_face_landmarks.append(landmark_flipped.reshape(10))

                current_image_array = np.asarray(current_face_images)
                current_landmark_array = np.asarray(current_face_landmarks)

                for j, img in enumerate(current_image_array):
                    if current_landmark_samples >= needed_landmark_samples:
                        break

                    # MEF: Why this check for landmarks falling within box?
                    # if np.sum(np.where(current_landmark_array[i] <= 0, 1, 0)) > 0 or \
                    #         np.sum(np.where(current_landmark_array[i] >= 1, 1, 0)) > 0:
                    #     pt.update_current_time()
                    #     continue

                    cv2.imwrite(join(landmark_dir, f"{generated_landmark_images}.jpg"), img)
                    fname = self._resize_and_save(landmark_dir, img, target_face_size, generated_landmark_images, "lm")
                    landmarks = [str(x) for x in list(current_landmark_array[i])]
                    landmark_file.write(fname + " -2 " + " ".join(landmarks) + "\n")
                    generated_landmark_images += 1
                    current_landmark_samples += 1

                # MEF: Reset the current_xxx arrays here so we don't go over them again!
                current_face_images = []
                current_face_landmarks = []

            processed_input_images = processed_input_images + 1
            if processed_input_images % 5000 == 0:
                mef.tsprint(f"\r{processed_input_images} / {total_number_of_input_images} images processed...")
            pt.update()

        landmark_file.close()
        return True
