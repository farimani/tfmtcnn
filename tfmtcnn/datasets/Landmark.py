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

import numpy as np
import cv2


def show_landmark(face, landmark):
    face_copied = face.copy().astype(np.uint8)
    for (x, y) in landmark:
        xx = int(face.shape[0] * x)
        yy = int(face.shape[1] * y)
        cv2.circle(face_copied, (xx, yy), 2, (0, 0, 0), -1)
    cv2.imshow("face_rot", face_copied)
    cv2.waitKey(0)


def rotate(img, bbox, landmark, alpha):
    center = ((bbox.left + bbox.right) / 2, (bbox.top + bbox.bottom) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, alpha, 1)
    img_rotated_by_alpha = cv2.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]))
    landmark_ = np.asarray([(rot_mat[0][0] * x + rot_mat[0][1] * y + rot_mat[0][2],
                             rot_mat[1][0] * x + rot_mat[1][1] * y + rot_mat[1][2])
                            for (x, y) in landmark])
    face = img_rotated_by_alpha[bbox.top:bbox.bottom + 1, bbox.left:bbox.right + 1]
    return face, landmark_


def flip(face, landmark):
    face_flipped_by_x = cv2.flip(face, 1)
    landmark_ = np.asarray([(1 - x, y) for (x, y) in landmark])
    landmark_[[0, 1]] = landmark_[[1, 0]]  # left eye<->right eye
    landmark_[[3, 4]] = landmark_[[4, 3]]  # left mouth<->right mouth
    return face_flipped_by_x, landmark_


def random_shift(landmark_gt, shift):
    diff = np.random.rand(5, 2)
    diff = (2 * diff - 1) * shift
    landmark_p = landmark_gt + diff
    return landmark_p


def random_shift_with_argument(landmark_gt, shift):
    # noinspection PyPep8Naming
    N = 2
    landmark_ps = np.zeros((N, 5, 2))
    for i in range(N):
        landmark_ps[i] = random_shift(landmark_gt, shift)
    return landmark_ps
