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
import numpy.random as npr

from tfmtcnn.utils.IoU import iou

import tfmtcnn.datasets.constants as datasets_constants
from tfmtcnn.datasets.DatasetFactory import DatasetFactory
from tfmtcnn.utils.convert_to_square import convert_to_square

import mef


class SimpleFaceDataset(object):
    __positive_parts = datasets_constants.positive_parts
    __part_parts = datasets_constants.partial_parts
    __negative_parts = datasets_constants.negative_parts

    def __init__(self, name='SimpleFaceDataset'):
        self._name = name
        self._pos_file = None
        self._part_file = None
        self._neg_file = None
        self._info_file = None
        self._pos_dir = ""
        self._part_dir = ""
        self._neg_dir = ""

        # MEF: 20 is a good trade-off. We get the speed with very few samples sacrificed for positives (~1/1000 faces)
        self._base_num_trys = 20  # 5000

        # MEF: Use the following metrics to get a reasonable number for base_number_of_attempts.
        self._skipped_negs = 0
        self._skipped_parts = 0
        self._skipped_pos = 0
        self._zero_negs = 0
        self._zero_parts = 0
        self._zero_pos = 0

        # MEF: Avoid checking on directory existence for every image write by keeping track...
        self._dirs_set = set()

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
    def read(annotation_image_dir, annotation_file_name, face_dataset_name='WIDERFaceDataset'):
        dataset = None
        face_dataset = DatasetFactory.face_dataset(face_dataset_name)
        if face_dataset.read(annotation_image_dir, annotation_file_name):
            dataset = face_dataset.data()

        return dataset

    @staticmethod
    def _generated_samples(target_root_dir):
        # MEF: Replace these with a more robust line counter. readlines() generate extra lines in this code on
        #      windows due to all the old os.lineseps in there...

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

    def _prep_files(self, target_dir):
        self._pos_dir = os.path.join(target_dir, 'positive')
        self._part_dir = os.path.join(target_dir, 'part')
        self._neg_dir = os.path.join(target_dir, 'negative')

        mef.create_dir_if_necessary(self._pos_dir, raise_on_error=True)
        mef.create_dir_if_necessary(self._part_dir, raise_on_error=True)
        mef.create_dir_if_necessary(self._neg_dir, raise_on_error=True)

        self._pos_file = open(SimpleFaceDataset.positive_file_name(target_dir), 'a+')
        self._part_file = open(SimpleFaceDataset.part_file_name(target_dir), 'a+')
        self._neg_file = open(SimpleFaceDataset.negative_file_name(target_dir), 'a+')
        self._info_file = open(os.path.join(target_dir, "info.txt"), 'a+')

        self._skipped_negs = 0
        self._skipped_parts = 0
        self._skipped_pos = 0
        self._zero_negs = 0
        self._zero_parts = 0
        self._zero_pos = 0

    def _close_files(self):
        self._pos_file.close()
        self._part_file.close()
        self._neg_file.close()
        self._info_file.close(_)
        self._pos_file = self._part_file = self._neg_file = self._info_file = None

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

    def _try_save_rand_neg(self, image_fname, image, image_bboxes, target_size, n_negs_so_far, overlap_bbox=None):
        """ Generate a random negative crop box and if the IOU meets the criteria, save it.
            If bbox is specified, negatives are picked with some overlap with the box.
        """
        imheight, imwidth, _ = image.shape
        cropsize = npr.randint(target_size, min(imwidth, imheight) / 2)

        if overlap_bbox is None:
            cx, cy = npr.randint(0, (imwidth - cropsize)), npr.randint(0, (imheight - cropsize))
        else:
            bbwidth = overlap_bbox[2] - overlap_bbox[0] + 1
            bbheight = overlap_bbox[3] - overlap_bbox[1] + 1
            dx = npr.randint(max(-cropsize, -overlap_bbox[0]), bbwidth)
            dy = npr.randint(max(-cropsize, -overlap_bbox[1]), bbheight)
            cx = max(0, overlap_bbox[0] + dx)
            cy = max(0, overlap_bbox[1] + dy)

            if cx + cropsize > imwidth and cy + cropsize > imheight:
                return False

        cropbox = np.array([cx, cy, cx + cropsize - 1, cy + cropsize - 1])
        maxiou = np.max(iou(cropbox, image_bboxes))
        iou_thresh = datasets_constants.abs_negative_iou if overlap_bbox is None else datasets_constants.negative_iou

        if maxiou < iou_thresh:
            cropped = image[cy:cy + cropsize, cx:cx + cropsize, :]
            fname = self._resize_and_save(self._neg_dir, cropped, target_size, n_negs_so_far, "simple-negative")
            self._neg_file.write(fname + ' 0\n')
            # record where it comes from for debugging...
            self._info_file.write(f"{mef.basename(fname)} <-- {image_fname}: ({cx},{cy}) ({cropsize}x{cropsize})\n")
            return True

        return False

    def _save_rand_negs(self, num_to_save, image_fname, image, image_bboxes, target_size,
                        n_negs_so_far, overlap_bbox=None):
        """ Save the given number of random negative crops to save. """
        n_saved = n_trys = 0
        max_trys = self._base_num_trys * num_to_save
        while n_saved < num_to_save and n_trys < max_trys:
            n_trys += 1
            if self._try_save_rand_neg(image_fname, image, image_bboxes, target_size,
                                       n_negs_so_far + n_saved, overlap_bbox):
                n_saved += 1

        # MEF: Use to find a good number of attempts...
        self._zero_negs += 1 if n_saved == 0 else 0
        self._skipped_negs += 1 if n_saved < num_to_save else 0
        return n_saved

    def _try_save_rand_pos_or_part(self, image_fname, image, target_size, n_pos_so_far, n_parts_so_far,
                                   need_pos, need_part, bbox):
        """ Generate a random crop box and if the IOU meets the criteria, save it as positive or parial.
            Return "pos", "part", or "" if the sample was saved as positive, partial or not saved.
        """
        imheight, imwidth, _ = image.shape

        # MEF: In practice, you get many situations where
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
        #      We pick a rectangular crop box, do the IOU calc, and once
        #      we find a good match, we convert the crop box to square.
        #      Even then, we will have cases that making the box square will always put the crop box
        #      extending beyond image boundaries.

        bbwidth, bbheight = bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1

        crop_w = npr.randint(int(bbwidth * 0.8), np.ceil(bbwidth * 1.25))
        crop_h = npr.randint(int(bbheight * 0.8), np.ceil(bbheight * 1.25))

        dx = npr.randint(-1.0 * bbwidth, +1.0 * bbwidth + 1) * 0.2
        dy = npr.randint(-1.0 * bbheight, +1.0 * bbheight + 1) * 0.2

        cx1 = int(max((bbox[0] + bbwidth / 2.0 + dx - crop_w / 2.0), 0))
        cy1 = int(max((bbox[1] + bbheight / 2.0 + dy - crop_h / 2.0), 0))
        cx2, cy2 = cx1 + crop_w - 1, cy1 + crop_h - 1

        if cx2 >= imwidth or cy2 >= imheight:
            return ""

        crop_box = np.array([cx1, cy1, cx2, cy2])
        cx1, cy1, cx2, cy2 = convert_to_square(crop_box.reshape(1, -1))[0]  # now convert to square

        if cx1 < 0 or cy1 < 0 or cx2 >= imwidth or cy2 >= imheight:  # out of bounds?
            return ""

        iou_amount = iou(crop_box, bbox.reshape(1, -1))
        ret = ""
        if (iou_amount >= datasets_constants.positive_iou and need_pos) or \
                (iou_amount >= datasets_constants.part_iou and need_part):
            cropsize = cx2 - cx1 + 1
            offset_x1, offset_y1 = (bbox[0] - cx1) / cropsize, (bbox[1] - cy1) / cropsize
            offset_x2, offset_y2 = (bbox[2] - cx2) / cropsize, (bbox[3] - cy2) / cropsize
            cropped = image[cy1:cy2 + 1, cx1:cx2 + 1, :]

            if iou_amount >= datasets_constants.positive_iou and need_pos:
                fname = self._resize_and_save(self._pos_dir, cropped, target_size, n_pos_so_far, "simple-positive")
                self._pos_file.write(f"{fname} 1 {offset_x1:.2f} {offset_y1:.2f} {offset_x2:.2f} {offset_y2:.2f}\n")
                ret = "pos"
            else:
                assert need_part and iou_amount >= datasets_constants.part_iou
                fname = self._resize_and_save(self._part_dir, cropped, target_size, n_parts_so_far, "simple-part")
                self._part_file.write(f"{fname} -1 {offset_x1:.2f} {offset_y1:.2f} {offset_x2:.2f} {offset_y2:.2f}\n")
                ret = "part"

            # record where it comes from for debugging...
            self._info_file.write(f"{mef.basename(fname)} <-- {image_fname}: ({cx1},{cy1}) ({cropsize}x{cropsize})\n")

        return ret

    def _save_rand_pos_and_parts(self, n_pos_to_save, n_parts_to_save, image_fname, image, target_size,
                                 n_pos_so_far, n_parts_so_far, bbox):
        n_pos_saved = n_parts_saved = n_trys = 0
        max_trys = self._base_num_trys * (n_pos_to_save + n_parts_to_save)
        while (n_pos_saved < n_pos_to_save or n_parts_saved < n_parts_to_save) and n_trys < max_trys:
            n_trys += 1
            ret = self._try_save_rand_pos_or_part(image_fname, image, target_size,
                                                  n_pos_so_far + n_pos_saved, n_parts_so_far + n_parts_saved,
                                                  n_pos_saved < n_pos_to_save, n_parts_saved < n_parts_to_save, bbox)
            if ret == "pos":
                n_pos_saved += 1
            elif ret == "part":
                n_parts_saved += 1

        self._zero_pos += 1 if n_pos_saved == 0 else 0
        self._skipped_pos += 1 if n_pos_saved < n_pos_to_save else 0
        self._zero_parts += 1 if n_parts_saved == 0 else 0
        self._skipped_parts += 1 if n_parts_saved < n_parts_to_save else 0
        return n_pos_saved, n_parts_saved

    def generate_simple_samples(self, annotation_image_dir, annotation_file_name, base_number_of_images,
                                target_face_size, target_dir):
        dataset = self.read(annotation_image_dir, annotation_file_name)
        if dataset is None:
            return False

        self._prep_files(target_dir)

        image_fns, bboxes, total_faces = dataset['images'], dataset['bboxes'], dataset['number_of_faces']
        total_images = len(image_fns)
        n_gen_pos, n_gen_parts, n_gen_negs = self._generated_samples(target_dir)

        n_remaining_pos = base_number_of_images * datasets_constants.positive_parts - n_gen_pos
        n_remaining_parts = base_number_of_images * datasets_constants.partial_parts - n_gen_parts

        total_remaining_negs = base_number_of_images * datasets_constants.negative_parts - n_gen_negs
        n_remaining_negs_overlap = int(total_remaining_negs * datasets_constants.negatives_from_bbox_ratio)
        n_remaining_negs_no_overlap = total_remaining_negs - n_remaining_negs_overlap

        n_bboxes_processed = 0

        pt = mef.ProgressText(total_faces)
        # pt = mef.ProgressText(n_remaining_pos + n_remaining_parts +
        #                       n_remaining_negs_overlap + n_remaining_negs_no_overlap)
        for idx, img_fname in enumerate(image_fns):
            # if n_remaining_pos <= 0 and n_remaining_parts <= 0 and n_remaining_negs <= 0:
            #     break
            n_img_bboxes = len(bboxes[idx])
            img = cv2.imread(img_fname)
            if img is None:
                print(f"WARNING: Could not read image file {img_fname}. Skipping...")
                n_bboxes_processed += n_img_bboxes
                continue

            # img_bboxes = np.array(bboxes[idx], dtype=np.float32).reshape(-1, 4)
            img_bboxes = np.array(bboxes[idx], dtype=np.int)
            n_imgs_left = total_images - idx
            n_bboxes_left = total_faces - n_bboxes_processed

            # number of positives and partials to make
            n_pos_per_box = int(n_remaining_pos / n_bboxes_left) if n_bboxes_left > 0 else 0
            n_parts_per_box = int(n_remaining_parts / n_bboxes_left) if n_bboxes_left > 0 else 0

            # number of negatives to make for this image, with and without overlap. we calculate them separately
            # so that we get an even number of overlap ones per face...
            n_negs_overlap_per_box = n_remaining_negs_overlap / n_bboxes_left if n_bboxes_left > 0 else 0
            n_negs_overlap_this_img = int(n_negs_overlap_per_box * n_img_bboxes)
            n_negs_no_overlap_this_img = int(n_remaining_negs_no_overlap / n_imgs_left)

            n_pos_saved = n_parts_saved = n_negs_saved_overlap = 0

            # Generate negatives with no overlap
            n_negs_saved = self._save_rand_negs(n_negs_no_overlap_this_img, img_fname, img, img_bboxes,
                                                target_face_size, n_gen_negs)

            # Now go over the labeled boxes and generate the positives, partials and negatives per box
            for bidx, bbox in enumerate(img_bboxes):
                # Generate positives and partials
                pos, parts = self._save_rand_pos_and_parts(n_pos_per_box, n_parts_per_box, img_fname, img,
                                                           target_face_size, n_gen_pos + n_pos_saved,
                                                           n_gen_parts + n_parts_saved, bbox)
                # Generate negatives with overlap
                n_negs_this_box = int(n_negs_overlap_this_img / (n_img_bboxes - bidx))
                negs = self._save_rand_negs(n_negs_this_box, img_fname, img, img_bboxes, target_face_size,
                                            n_gen_negs + n_negs_saved, bbox)
                n_pos_saved += pos
                n_parts_saved += parts
                n_negs_saved += negs
                n_negs_saved_overlap += negs
                n_negs_overlap_this_img -= negs
                n_bboxes_processed += 1
                pt.update(f"{n_bboxes_processed}/{total_faces} faces")        # update the progress time and show

            n_gen_pos += n_pos_saved
            n_gen_parts += n_parts_saved
            n_gen_negs += n_negs_saved
            n_remaining_pos -= n_pos_saved
            n_remaining_parts -= n_parts_saved
            n_remaining_negs_no_overlap -= (n_negs_saved - n_negs_saved_overlap)
            n_remaining_negs_overlap -= n_negs_saved_overlap

            if (idx+1) % 1000 == 0:
                print(f"\r{idx+1}/{total_images} images processed - "
                      f"positives - {n_gen_pos},  partials - {n_gen_parts}, negatives - {n_gen_negs}")
                # MEF: Use the following metrics to get a reasonable number for base_number_of_attempts.
                # print(f"zero -ves: {self._zero_negs}, skipped -ves:{self._skipped_negs}")
                # print(f"zero +ves: {self._zero_pos}, skipped +ves:{self._skipped_pos}")
                # print(f"zero parts: {self._zero_parts}, skipped parts:{self._skipped_parts}")

        return True

    def generate_simple_samples0(self, annotation_image_dir, annotation_file_name, base_number_of_images,
                                 target_face_size, target_root_dir):
        dataset = self._read(annotation_image_dir, annotation_file_name)
        if dataset is None:
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

        negative_samples_per_image_ratio = (SimpleFaceDataset.__negative_parts - 1)
        needed_negative_samples = base_number_of_images - (generated_negative_samples /
                                                           SimpleFaceDataset.__negative_parts)
        needed_base_negative_samples = needed_negative_samples / number_of_faces
        needed_negative_samples_per_image = int(negative_samples_per_image_ratio * needed_base_negative_samples *
                                                (number_of_faces / len(image_file_names)))
        needed_negative_samples_per_image = max(0, needed_negative_samples_per_image)

        needed_negative_samples_per_bounding_box = np.ceil((SimpleFaceDataset.__negative_parts -
                                                            negative_samples_per_image_ratio) *
                                                           needed_base_negative_samples)
        needed_negative_samples_per_bounding_box = max(0, needed_negative_samples_per_bounding_box)
        needed_positive_samples = (base_number_of_images * SimpleFaceDataset.__positive_parts) - \
            generated_positive_samples
        needed_positive_samples_per_bounding_box = np.ceil(1.0 * needed_positive_samples / number_of_faces)
        needed_positive_samples_per_bounding_box = max(0, needed_positive_samples_per_bounding_box)
        needed_part_samples = (base_number_of_images * SimpleFaceDataset.__part_parts) - generated_part_samples
        needed_part_samples_per_bounding_box = np.ceil(needed_part_samples / number_of_faces)
        needed_part_samples_per_bounding_box = max(0, needed_part_samples_per_bounding_box)

        # MEF: 20 is a good trade-off. We get the speed with very few samples sacrificed for positives (~1/1000 faces)
        base_number_of_attempts = 20  # 5000
        current_image_number = 0

        # MEF: if you want to resume from an interrupted loop, set this appropriately. Make sure to interrupt at the
        # outer loop to keep things in order!
        skip_count = 0
        pt = mef.ProgressText(len(image_file_names) - skip_count)

        # MEF: Use the following metrics to get a reasonable number for base_number_of_attempts.
        skipped_negatives = skipped_partials = skipped_positives = 0
        zero_negatives = zero_partials = zero_positives = 0

        for i, image_file_path in enumerate(image_file_names, skip_count):
            bboxes = np.array(ground_truth_boxes[i], dtype=np.float32).reshape(-1, 4)
            current_image = cv2.imread(image_file_path)
            img_height, img_width, _ = current_image.shape

            needed_negative_samples = needed_negative_samples_per_image
            maximum_attempts = base_number_of_attempts * needed_negative_samples
            negative_images = number_of_attempts = 0
            while negative_images < needed_negative_samples and number_of_attempts < maximum_attempts:
                number_of_attempts += 1
                crop_size = npr.randint(target_face_size, min(img_width, img_height) / 2)
                nx, ny = npr.randint(0, (img_width - crop_size)), npr.randint(0, (img_height - crop_size))
                crop_box = np.array([nx, ny, nx + crop_size - 1, ny + crop_size - 1])
                current_iou = iou(crop_box, bboxes)

                if np.max(current_iou) < DatasetFactory.negative_iou():
                    cropped_image = current_image[ny:ny + crop_size, nx:nx + crop_size, :]
                    resized_image = mef.resize_image(cropped_image, target_face_size, target_face_size)
                    file_path = os.path.join(negative_dir, f"simple-negative-{generated_negative_samples}.jpg")
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
            for bounding_box in bboxes:
                x1, y1, x2, y2 = bounding_box
                if x1 < 0 or y1 < 0:
                    continue

                bounding_box_width = x2 - x1 + 1
                bounding_box_height = y2 - y1 + 1
                maximum_attempts = base_number_of_attempts * needed_negative_samples
                negative_images = number_of_attempts = 0
                while negative_images < needed_negative_samples and number_of_attempts < maximum_attempts:
                    number_of_attempts += 1
                    crop_size = npr.randint(target_face_size, min(img_width, img_height) / 2)
                    delta_x = npr.randint(max(-1 * crop_size, -1 * x1), bounding_box_width)
                    delta_y = npr.randint(max(-1 * crop_size, -1 * y1), bounding_box_height)
                    nx1 = int(max(0, x1 + delta_x))
                    ny1 = int(max(0, y1 + delta_y))
                    if (nx1 + crop_size) > img_width or (ny1 + crop_size) > img_height:
                        continue

                    crop_box = np.array([nx1, ny1, nx1 + crop_size - 1, ny1 + crop_size - 1])
                    current_iou = iou(crop_box, bboxes)

                    if np.max(current_iou) < DatasetFactory.negative_iou():
                        cropped_image = current_image[ny1:ny1 + crop_size, nx1:nx1 + crop_size, :]
                        resized_image = mef.resize_image(cropped_image, target_face_size, target_face_size)
                        file_path = os.path.join(negative_dir, f"simple-negative-{generated_negative_samples}.jpg")
                        negative_file.write(file_path + ' 0\n')
                        cv2.imwrite(file_path, resized_image)
                        generated_negative_samples += 1
                        negative_images += 1

                if negative_images == 0:
                    zero_negatives += 1
                elif negative_images < needed_negative_samples:
                    skipped_negatives += 1

                maximum_attempts = base_number_of_attempts * (needed_positive_samples + needed_part_samples)
                positive_images = part_images = number_of_attempts = 0

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
                    nx2, ny2 = nx1 + crop_w - 1, ny1 + crop_h - 1

                    if nx2 >= img_width or ny2 >= img_height:
                        update_progress(False)
                        continue

                    crop_box = np.array([nx1, ny1, nx2, ny2])
                    nx1, ny1, nx2, ny2 = convert_to_square(crop_box.reshape(1, -1))[0]  # now convert to square

                    if nx1 < 0 or ny1 < 0 or nx2 >= img_width or ny2 >= img_height:   # out of bounds?
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
                        cropped_image = current_image[ny1:ny2+1, nx1:nx2+1, :]
                        resized_image = mef.resize_image(cropped_image, target_face_size, target_face_size)
                        if iou_amount >= DatasetFactory.positive_iou() and positive_images < needed_positive_samples:
                            file_path = os.path.join(positive_dir, f"simple-positive-{generated_positive_samples}.jpg")
                            positive_file.write(f"{file_path} 1 {offset_x1:.2f} {offset_y1:.2f} "
                                                f"{offset_x2:.2f} {offset_y2:.2f}\n")
                            cv2.imwrite(file_path, resized_image)
                            generated_positive_samples += 1
                            positive_images += 1
                        else:
                            assert part_images < needed_part_samples and iou_amount >= DatasetFactory.part_iou()
                            file_path = os.path.join(part_dir, f"simple-part-{generated_part_samples}.jpg")
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


