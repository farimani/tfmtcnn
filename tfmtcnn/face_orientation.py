#######################################################################################
#
# face_orientation.py
# Project General
#
# FaceOrientation class implements head pose estimates, partly based on the
# following paper:
#
#   DETERMINING THE GAZE OF FACES IN IMAGES
#       A. H. Gee and R. Cipolla
#
#   http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.28.5324&rep=rep1&type=pdf
#
# Created by mehran on 03 / 21 / 19.
# Copyright © 2019 Percipo Inc. All rights reserved.
#
#######################################################################################

import sys
import math
import numpy as np

import mef

_CNTR_TO_CRNR_RATIO = .697          # eye center distance to eye outer corner distance ratio
_LE_TO_LF_RATIO = .95               # eye distance to nose_bridge-mouth distance ratio

_REL_FRAME_EYE_DIST = (51 - 21) / 72
_REL_FRAME_NOSE_BRIDGE_X = .5
_REL_FRAME_NOSE_BRIDGE_Y = 21 / 72

_LE_TO_LF_RATIO_2 = _LE_TO_LF_RATIO / 2
_REL_FRAME_EYE_DIST_2 = _REL_FRAME_EYE_DIST * 2
_REL_FRAME_NB_TO_MOUTH_DIST = _REL_FRAME_EYE_DIST / _LE_TO_LF_RATIO
_REL_FRAME_NB_TO_MOUTH_DIST_X_2 = _REL_FRAME_EYE_DIST * 2 / _LE_TO_LF_RATIO

# lower limit of sine of angle between projected orthogonal vectors for projection
# to be considered as sufficiently orientation preserving; must be >= 0.
_PROJ_SINE_LL = .5

_PROJ_SINE_LL2 = _PROJ_SINE_LL * _PROJ_SINE_LL


class FaceOrientation:
    def __init__(self, leye, reye, nose_tip, lmouth, rmouth):
        # landmarks
        self._leye = leye
        self._reye = reye
        self._nose_tip = nose_tip
        self._lmouth = lmouth
        self._rmouth = rmouth

        # orientation info
        self._est_center = None  # this is the same as nose base and different from the center straight from landmarks
        self._normal = None      # normal vector
        self._est_size = None    # an indication of the size
        self._up_down = 0        # angles...
        self._left_right = 0
        self._left_right_adjusted = 0   # this one has a hacky adjustment to compensate for LR insensitivity
        self._in_plane = 0

        # points and vectors that we calculate to derive orientation:
        self._mid_eyes = None
        self._mid_mouth = None
        self._nose_base = None
        self._sym_vec = None
        self._nose_vec = None
        self._sym_vec_at_min = None
        self._4pt_center = None                     # this center is the average of 4 eye and mouth points
        self._LR_fudge_ratio = 0                    # a measure of how much to fudge L/R based values. see below...
        self._is_looking_to_their_right = False     # Flag set if subject is looking towards their right shoulder
        self._rot_pts = None

        # cached stuff
        self._rect = None
        self._fo_rect = None

        self._calc_frontal_points_and_vectors()     # And calculate and initialize a bunch of things!

    def _calc_frontal_points_and_vectors(self):
        """ Calculate and set: mid_eyes, mid_mouth, nose_base, sym_vec, nose_vec, sym_vec_at_min
            Also calculates the inplane angle and the LR fudge ratio.
        """
        # Dynamically adjust ratio_m based on estimated expression
        mouth_diff = np.subtract(self._rmouth, self._lmouth)
        eye_diff = np.subtract(self._reye, self._leye)

        # relativeMouthWidth = mouthDiff.magnitude() / eyeDiff.magnitude();
        relative_mouth_width = np.linalg.norm(mouth_diff) / np.linalg.norm(eye_diff)

        # ratio_m = 0.5
        # if relative_mouth_width < 0.55:
        #     ratio_m = 0.55
        #
        # if relative_mouth_width > 0.7:
        #     ratio_m = 0.45

        # If the relative mouth width is < 0.55, use ratio of 0.55
        # If it is > 1.25, use ratio of 0.35
        # Otherwise it's a straight line between these two points
        ratio_at_min, ratio_at_max = 0.5, 0.3
        min_pt, max_pt = [0.55, ratio_at_min], [1.25, ratio_at_max]

        if relative_mouth_width < min_pt[0]:
            ratio_m = ratio_at_min
        elif relative_mouth_width > max_pt[0]:
            ratio_m = ratio_at_max
        else:
            m = (min_pt[1]-max_pt[1]) / (min_pt[0]-max_pt[0])
            b = min_pt[1] - m * min_pt[0]
            ratio_m = m * relative_mouth_width + b

        # Get 2D facial symmetry vector and 2D nose vector
        self._mid_eyes = np.add(self._leye, self._reye) * 0.5
        self._mid_mouth = np.add(self._lmouth, self._rmouth) * 0.5
        self._sym_vec = np.subtract(self._mid_eyes, self._mid_mouth)
        self._nose_base = np.add(self._mid_mouth, self._sym_vec * ratio_m)
        self._nose_vec = np.subtract(self._nose_tip, self._nose_base)

        # estimated face symmetry vector at min relative mouth width
        # assert ratio_m < 1
        self._sym_vec_at_min = self._sym_vec * (1 - ratio_m) / (1 - ratio_at_min)

        h_vector = (-self._sym_vec[1], self._sym_vec[0])
        self._in_plane = 90 - mef.angle_between_vectors(h_vector, (0, 1))

        # average of the 4 eye/mouth points
        self._4pt_center = np.add(np.add(self._lmouth, self._rmouth), np.add(self._leye, self._reye)) * 0.25

        # Now rotate the landmarks around the nose base and calculate the mouth overhang
        # We use this as a secondary measure of the extent of L/R rotation, and we correct a couple of things
        # based on it. First, we correct the L/R angle which becomes insensitive to high amounts of rotations in
        # Cipolla's original implementation. Secondly, we use it to move the center of the face rectangle so as
        # to get a tighter crop around the face when the face is rotated.
        #
        # The way it works is that as the face rotates L/R, the interoccular distance becomes small and thus
        # the mouth corners fall out of the rectangle estimated based on eye centers.
        #
        rot_pts = mef.rotate_points(np.array([self._leye, self._reye, self._nose_tip, self._lmouth, self._rmouth]),
                                    self._in_plane, self._4pt_center)
        r = self._calc_rect_from_eyes(rot_pts[0], rot_pts[1])
        max_mouth_y = max(rot_pts[3][1], rot_pts[4][1])
        r[3] = (r[1] + ((r[3] - r[1] + 1) * 0.85) - 1)       # MEF: HEREHERE
        mouth_overhang = max_mouth_y - r[3]
        self._LR_fudge_ratio = 0.8 * mouth_overhang / (r[3] - r[1] + 1) if mouth_overhang > 0 else 0

        # Flag set if subject is looking towards their right shoulder
        self._is_looking_to_their_right = abs(rot_pts[2][0] - rot_pts[0][0]) < abs(rot_pts[2][0] - rot_pts[1][0])
        self._rot_pts = rot_pts

    @staticmethod
    def _calc_center_from_eyes(leye, reye):
        """ Calculate face center from eye centers only. """
        mid_eyes = np.add(leye, reye) * 0.5
        diff = np.subtract(leye, reye)
        run, rise = diff[0], diff[1]

        if run > 0:
            run = -run
            rise = -rise

        ratio = 0.5
        center = (mid_eyes[0] + (rise * ratio), mid_eyes[1] - (run * ratio))
        return center

    @staticmethod
    def _calc_size_from_eyes(leye, reye):
        """ Calculate size of the face from eye centers only. """
        dist_between = np.linalg.norm(np.subtract(leye, reye))
        width = dist_between * 72 / 30
        return width, width

    @classmethod
    def _calc_rect_from_eyes(cls, leye, reye):
        """ Calculate the face rectangle based on eye centers only. """
        size = cls._calc_size_from_eyes(leye, reye)
        center = cls._calc_center_from_eyes(leye, reye)
        rx, ry = center[0] - size[0] / 2, center[1] - size[1] / 2
        return [rx, ry, rx + size[0] - 1, ry + size[1] - 1]

    @property
    def est_center(self):
        return self._est_center

    @property
    def normal(self):
        return self._normal

    @property
    def est_size(self):
        return self._est_size

    @property
    def up_down(self):
        return self._up_down

    @property
    def left_right(self):
        return self._left_right

    @property
    def left_right_adjusted(self):
        return self._left_right_adjusted

    @property
    def in_plane(self):
        return self._in_plane

    def calc_center_from_eyes(self):
        """ Calculate face center from eye centers only. """
        return self._calc_center_from_eyes(self._leye, self._reye)

    def calc_size_from_eyes(self):
        """ Calculate size of the face from eye centers only. """
        return self._calc_size_from_eyes(self._leye, self._reye)

    def calc_rect_from_eyes(self):
        """ Calculate the face rectangle based on eye centers only. """
        return self._calc_rect_from_eyes(self._leye, self._reye)

    def calc_center(self):
        """ Calculate center of face based on eye centers and mouth corners. """
        # sum_points = np.add(np.add(self._lmouth, self._rmouth), np.add(self._leye, self._reye))
        # return sum_points * 0.25
        return self._4pt_center

    def calc_size(self, calc_left_to_right_dir=False):
        """
         Calculate the size of the face rectangle based on eye centers and mouth corners.
         Nik's algo.

         If calc_left_to_right_dir is set, 'left_to_right_dir' will be set to a unit vector
         u = [ u1 u2] that represents the direction from left to right in the face, (as seen from a
         viewer sharing its vertical axis.)
         More precisely, u is chosen so as to minimize

         < u, a>^2 + < v, d>^2

         where v is the orthogonal unit vector [ -u2 u1], d is the eye vector from the left to the right eye,
         & a is the symmetry vector from the nose bridge to the center of the mouth times '_LE_TO_LF_RATIO'.
         The constant '_LE_TO_LF_RATIO' is chosen so as to equalize the magnitude of the vectors a and d
         for the typical frontal face.

         The return value is the width of an appropriate bounding box.
         This is found by using the vectors d & a to compute the 3D distance (in pixels) between the eyes,
         and then multiplying this distance by a constant.
         The constant is set to

         1 / '_REL_FRAME_EYE_DIST' = 72 / ( 51 - 21) = 2.4

         in order to comply with the average frontal face from our data set.
         The returned width is <= 0 if & only if the left & right eyes coincide.
         In this case '*leftToRightP' has no meaning, & will be set to zero.

         The average face from our data set and face detector has eyes at 21,21 and
         51,21 (72x72 face, 30 dist between eyes)

         This function agrees with that for frontal faces when '_LE_TO_LF_RATIO_2' = 1.
        """
        # PGPoint     a2;             // eye corner sum to mouth corner sum vec
        # PGPoint     mouthCenter2;   // mouth corner sum vec
        # PGPoint     noseBridge2;    // eye corner sum vec
        # float       J11;            // d"srcX" / d"unwarpedX" smart warp
        # float       J12;            // d"srcX" / d"unwarpedY" smart warp
        # float       J21;            // d"srcY" / d"unwarpedX" smart warp
        # float       J22;            // d"srcY" / d"unwarpedY" smart warp
        # float       J11_2;          // 'J11 * J11'
        # float       J12_2;          // 'J12 * J12'
        # float       J21_2;          // 'J21 * J21'
        # float       J22_2;          // 'J22 * J22'
        # float       detJ;           // det 'J'
        # float       eye2;           // < eye, eye>
        # float       sym2;           // < sym, sym>
        # float       faceWidth;      // face rect width

        leye, reye, lmouth, rmouth = self._leye, self._reye, self._lmouth, self._rmouth

        mouth_center2 = np.add(lmouth, rmouth)  # mouth corners sum vec
        nose_bridge2 = np.add(leye, reye)       # eye centers sum vec
        a2 = np.subtract(mouth_center2, nose_bridge2)       # eye centers sum to mouth corners sum vec
        jacobian = np.empty((2, 2))

        # j11 = reye[0] - leye[0]
        # j21 = reye[1] - leye[1]
        # j12 = a2[0] * _LE_TO_LF_RATIO_2
        # j22 = a2[1] * _LE_TO_LF_RATIO_2

        jacobian[:, 0] = np.subtract(reye, leye)
        jacobian[:, 1] = a2 * _LE_TO_LF_RATIO_2
        det_j = np.linalg.det(jacobian)

        jacobian2 = jacobian**2
        eye2 = jacobian2[0, 0] + jacobian2[1, 0]
        sym2 = jacobian2[0, 1] + jacobian2[1, 1]

        # Below we check that "sin( alpha)" where alpha is the angle between the eye &
        # symmetry vectors is >= '_PROJ_SINE_LL' (> 0.)
        # If so, the estimated mouth corners will be deemed unreliable, & thence
        # not used.
        # Instead the symmetry vector will be assumed to be orthogonal to the eye vector,
        # which then solely determines the in-plane rotation of a believed to be perfectly frontal face.

        left_to_right_dir = None
        if det_j <= 0 or det_j**2 <= eye2 * sym2 * _PROJ_SINE_LL2:       # mouth corners unreliable
            face_width = math.sqrt(eye2) / _REL_FRAME_EYE_DIST

            if calc_left_to_right_dir:      # orientation request
                if face_width != 0:         # valid eye vector
                    left_to_right_dir = jacobian[:, 0] / face_width
                else:                       # zero eye vector
                    left_to_right_dir = np.zeros(2)
        else:                                                               # mouth corners reliable
            mag_square_sum2 = (eye2 + sym2) / 2
            mag_square_dif2 = (eye2 - sym2) / 2
            i_prod = np.inner(jacobian[:, 0], jacobian[:, 1])
            j_mag = mag_square_sum2 + math.sqrt(mag_square_dif2**2 + i_prod**2)
            face_width = math.sqrt(j_mag) / _REL_FRAME_EYE_DIST

            if calc_left_to_right_dir:      # orientation request
                x_rot = jacobian2[0, 0] + jacobian2[1, 1] - jacobian2[0, 1] - jacobian2[1, 0]
                y_rot = (jacobian[0, 0] * jacobian[1, 0] - jacobian[0, 1] * jacobian2[1, 1]) * 2
                r_rot = math.sqrt(x_rot*2 + y_rot**2)       # 'detJ > 0' => 'rRot > 0'
                c_rot = math.sqrt((1 + x_rot / r_rot) * .5)
                s_rot = y_rot / (r_rot * c_rot * 2)
                left_to_right_dir = (c_rot, s_rot)

        return (face_width, face_width), left_to_right_dir

    def calc_rect(self):
        """
        Calculate face rectangle based on eye centers and mouth corners.
        Nik's algo.

        """

        if self._rect is None:
            (w, h), _ = self.calc_size()
            cx, cy = self._4pt_center

            if self._LR_fudge_ratio > 0:
                center_shift = min(w * 0.2, self._LR_fudge_ratio * w * 10)
                # cx, cy = mef.rotate_point((cx, cy), self._in_plane, self._4pt_center)
                if self._is_looking_to_their_right:
                    cx += center_shift
                else:
                    cx -= center_shift

                cx, cy = mef.rotate_point((cx, cy), -self._in_plane, self._4pt_center)

            rx, ry = cx - w / 2, cy - h / 2
            self._rect = [rx, ry, rx + w - 1, ry + h - 1]

        return self._rect

    def calc_orienation(self):
        """ Calculate face orientation more or less based on the paper:

            DETERMINING THE GAZE OF FACES IN IMAGES
            A. H. Gee and R. Cipolla

            Note: _calc_frontal_points_and_vectors() must be called already.

        """
        if self._fo_rect is not None:
            return self._fo_rect

        ratio_n = 0.6
        theta = mef.angle_between_vectors(self._sym_vec, self._nose_vec, degrees=False)
        tau = mef.angle_between_vectors((1, 0), self._nose_vec, degrees=False)

        if self._nose_vec[1] < 0:
            tau = (2.0 * math.pi) - tau

        # Find image measurement 1
        sym_vec_magnitude = np.linalg.norm(self._sym_vec)
        m1 = np.linalg.norm(self._nose_vec) / sym_vec_magnitude
        m1 *= m1
        nose_ratio_squared = ratio_n * ratio_n

        # Calculate face normal
        if mef.vectors_are_almost_parallel(self._sym_vec, self._nose_vec):      # theta == 0 or PI
            # Nose vector lies practically on top of symmetry vector
            # Paper's method + mehran's adds ...
            # First calculate dz (z component of the unit normal)
            z_squared = nose_ratio_squared / (m1 + nose_ratio_squared)

            # nose vector and symvector align or almost align (m2 == 1).
            # so face normal would have an almost 0 x component
            # derive y component from z
            self._normal = (0, math.sqrt(1.0 - z_squared), math.sqrt(z_squared))  # z component
        else:
            # Find image measurement 2
            cos_theta = math.cos(theta)
            m2 = cos_theta**2

            # Solve quadratic
            a_term = (1.0 - m2) * nose_ratio_squared
            b_term = m1 - nose_ratio_squared + (2.0 * m2 * nose_ratio_squared)
            c_term = - (m2 * nose_ratio_squared)
            z_squared1 = (-b_term + math.sqrt(b_term**2 - (4.0 * a_term * c_term))) / (2.0 * a_term)

            # The two roots are either both zero, or only the greater is positive.
            # Our solution is therefore always the square root of the greater of
            # the two roots of the quadratic equation.
            z_squared1 = max(z_squared1, 0)  # MEF: protect against numeric error (< 0)
            z = math.sqrt(z_squared1)
            z = min(z, 1)                    # MEF: protect against numeric error (acos valid -1 to 1 only)
            phi = math.acos(z)

            # Normal vector can now be derived
            sin_phi = math.sin(phi)
            self._normal = (sin_phi * math.cos(tau), sin_phi * math.sin(tau), -z)

        # Calculate rotation angles using normal vector
        # h_vector = (-self._sym_vec[1], self._sym_vec[0])
        h_vector_3d = (-self._sym_vec[1], self._sym_vec[0], 0)
        sym_vec_3d = (self._sym_vec[0], self._sym_vec[1], 0)

        sym_to_normal_angle = mef.angle_between_vectors(sym_vec_3d, self._normal, degrees=False)
        self._up_down = 90 - math.degrees(sym_to_normal_angle)
        self._left_right = 90 - mef.angle_between_vectors(h_vector_3d, self._normal)
        # self._in_plane = 90 - mef.angle_between_vectors(h_vector, (0, 1))   # already calculated...

        # Finally estimate the centre and calculate some measure of size
        eye_vector = np.subtract(self._reye, self._leye)
        eye_vector_3d = (eye_vector[0], eye_vector[1], 0)
        eye_to_normal_angle = mef.angle_between_vectors(eye_vector_3d, self._normal, degrees=False)

        size_w = np.linalg.norm(eye_vector) / math.sin(eye_to_normal_angle) + \
            sym_vec_magnitude / math.sin(sym_to_normal_angle)

        self._est_size = (size_w, size_w)

        # Hack to compensate for L/R insensitivity
        if self._LR_fudge_ratio <= 0:
            self._left_right_adjusted = self._left_right
            self._est_center = self._nose_base
        else:
            angle_fudge = self._LR_fudge_ratio * 90
            center_shift = min(size_w * 0.2, self._LR_fudge_ratio * size_w * 10)
            center = mef.rotate_point(self._nose_base, self._in_plane, self._4pt_center)

            if self._left_right < 0:
                self._left_right_adjusted = max(self._left_right - angle_fudge, -90)
                center[0] += center_shift
            else:
                self._left_right_adjusted = min(self._left_right + angle_fudge, 90)
                center[0] -= center_shift

            self._est_center = mef.rotate_point(center, -self._in_plane, self._4pt_center)

        fx, fy = self._est_center[0] - self._est_size[0] / 2, self._est_center[1] - self._est_size[1] / 2
        self._fo_rect = [fx, fy, fx + self._est_size[0] - 1, fy + self._est_size[1] - 1]
        return self._fo_rect

    def tight_crop(self):
        """
        Return a tigher crop rectangle using heuristics to pick from Cipollas and nik's rect estimates.
        Bit hoaky but works pretty well.

        :return:
        """
        rect = self.calc_rect()
        abs_inplane = abs(self._in_plane)
        abs_lr = abs(self._left_right_adjusted)

        if abs_inplane < 40:
            fo_rect = self.calc_orienation()

            if abs_lr > 7:
                alpha = min(abs_lr, 15) / 15
                one_minus = 1 - alpha

                if self._is_looking_to_their_right:
                    x1, x2 = fo_rect[0] * alpha + one_minus * rect[0], rect[2] * alpha + one_minus * fo_rect[2]
                else:
                    x1, x2 = rect[0] * alpha + one_minus * fo_rect[0], fo_rect[2] * alpha + one_minus * rect[2]
            else:
                x1, x2 = (fo_rect[0] + rect[0]) / 2, (fo_rect[2] + rect[2]) / 2

            y1, y2 = rect[1], rect[3]       # we use the height from calc_rect() rectangle.
            rect = [x1, y1, x2, y2]
        elif abs_inplane > 75:
            pts = self.rot_crop()
            rect = mef.rect_hull(pts)

        return rect

    def rot_crop(self):
        """
        Get the rotated crop of the face.

        :return: Set of 4 points defining the polygon for the rotated face.
        """
        fo = FaceOrientation(self._rot_pts[0], self._rot_pts[1], self._rot_pts[2], self._rot_pts[3], self._rot_pts[4])
        rect = fo.calc_rect()
        fo.calc_orienation()
        ctr, size = fo.est_center, fo.est_size
        fx1, fy1 = ctr[0] - size[0] / 2, ctr[1] - size[1] / 2
        fo_rect = [fx1, fy1, fx1 + size[0] - 1, fy1 + size[1] - 1]

        if fo._is_looking_to_their_right:
            x1, x2 = fo_rect[0], rect[2]
        else:
            x1, x2 = rect[0], fo_rect[2]

        y1, y2 = rect[1], rect[3]       # we use the height from calc_rect() rectangle.

        # get the 4 pts of the rectangle now
        pts4 = np.array([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
        rot_pts = mef.rotate_points(pts4, -self._in_plane, self._4pt_center)
        return rot_pts

    def nik_crop(self):
        """
        Nik's experiments for getting a tight crop. Not done yet...
        MEF: TODO: Finish!
        :return:
        """
        pa = 1          # sym vec param
        pc = 0.4       # center param
        pr = 1.1        # radius param
        sym_vec_scaled = self._sym_vec * pa
        sym_vec_scaled_mag2 = np.linalg.norm(sym_vec_scaled)**2
        eye_vector = np.subtract(self._reye, self._leye)
        eye_vector_mag2 = np.linalg.norm(eye_vector)**2

        # intermediate quantities
        q_sum = (eye_vector_mag2 + sym_vec_scaled_mag2) / 2
        q_diff = (eye_vector_mag2 - sym_vec_scaled_mag2) / 2
        q_inner = np.dot(eye_vector, sym_vec_scaled)
        q_root = math.sqrt(q_diff**2 + q_inner**2)

        if q_root <= sys.float_info.epsilon:
            projected_normal = np.array([0, 0])
        else:
            eigen_v = np.array([q_inner, q_root - q_diff]) if q_diff < 0 else np.array([q_root + q_diff, q_inner])
            eigen_v = eigen_v / np.linalg.norm(eigen_v)
            projected_normal = (eigen_v[0] * eye_vector + eigen_v[1] * sym_vec_scaled) * \
                math.sqrt(2 * q_root / (q_sum + q_root))
            # take the orthogonal vector
            projected_normal = np.array([-projected_normal[1], projected_normal[0]])

        same_direction = np.dot(self._nose_vec, projected_normal) >= 0

        if same_direction:
            center = self._4pt_center - (projected_normal * pc)
        else:
            center = self._4pt_center + (projected_normal * pc)

        radius = math.ceil(math.sqrt(q_sum + q_root) * pr)
        return [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]





