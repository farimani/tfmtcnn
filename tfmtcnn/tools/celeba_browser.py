#######################################################################################
#
# celeba_browser.py
# Project MTCNN
#
# GUI browser for CelebA dataset
#
# Created by mehran on 03 / 20 / 19.
# Copyright © 2019 Percipo Inc. All rights reserved.
#
#######################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import time
import re
import argparse

import cv2

import mef
import face_orientation as forient
import browser_gui

WIN_TITLE = "CelebA Browser"


HELP_INFO = [("'h'",                        "This help screen"),
             (),
             ("Arrows, 'p', 'n', ' '",      "Navigate"),
             ("'j'",                        "Jump to an image"),
             (),
             ("'l'",                        "Toggle marked bounding box"),
             ("'l'",                        "Toggle landmarks"),
             ("'b'",                        "Toggle extra bounding boxes"),
             ("'r'",                        "Toggle rotated cropped box"),
             (),
             ("ESC or 'q'",                 "Quit")]


class CelebABrowser(browser_gui.BrowserGUI):
    def __init__(self):
        super().__init__(WIN_TITLE)

    def draw_bbox(self, img, bb_info, scale=1.0, color=(20, 255, 20)):
        bb = [int(float(bb_info[x]) * scale) for x in range(1, 5)]
        bb[2] = bb[0] + bb[2] - 1       # convert from width/height to x2,y2
        bb[3] = bb[1] + bb[3] - 1
        self.draw_rect(img, bb, color=color)

    def draw_landmarks(self, img, lm_info, scale=1.0):
        lcolor, mcolor, rcolor = (20, 20, 255), (255, 255, 100), (255, 100, 255)
        colors = [lcolor, rcolor, mcolor, lcolor, rcolor]
        lm = [int(float(lm_info[x]) * scale) for x in range(1, 11)]

        for i in range(0, 10, 2):
            self.draw_circle(img, lm[i], lm[i+1]-1, 2, color=(10, 10, 10), thick=cv2.FILLED)
            self.draw_circle(img, lm[i], lm[i+1]+1, 2, color=(255, 255, 255), thick=cv2.FILLED)
            self.draw_circle(img, lm[i], lm[i+1], 2, color=colors[i//2], thick=cv2.FILLED)

    def draw_info(self, img, info_top, info_bottom=None):
        font_saved = self.get_font()
        fscale = 0.4
        self.set_font(scale=fscale, thick=1)
        color = (255, 255, 230)
        tx, ty = (10, 15)
        self.put_text(img, info_top, tx, ty, color=color)
        self.put_text(img, info_top, tx+1, ty, color=(0, 0, 0))

        if info_bottom is not None:
            self.set_font(scale=fscale * 0.8)
            tw, th, baseline = self.get_text_size(info_bottom)
            tx, ty = 10, img.shape[0] - th - baseline
            self.put_text(img, info_bottom, tx, ty, color=color)
            self.put_text(img, info_bottom, tx + 1, ty, color=(0, 0, 0))

        self.set_font(*font_saved)

    def load_and_render_image(self, fname, bb_info, lm_info, image_num, total_images,
                              show_bbox, show_landmarks, show_bbox_from_landmarks, show_rot_bbox):
        info = f"{image_num} / {total_images} - {mef.basename(fname)}"
        info_bottom = ""
        img, scale = self.imread_and_scale(fname)

        if img is None:
            img = self.msg_img(f"ERROR: Can't read image {mef.basename(fname)}")
            info += " INVALID"
        else:
            if show_bbox:
                self.draw_bbox(img, bb_info, scale=scale)

            leye, reye = (float(lm_info[1]), float(lm_info[2])), (float(lm_info[3]), float(lm_info[4]))
            nose_tip = (float(lm_info[5]), float(lm_info[6]))
            lmouth, rmouth = (float(lm_info[7]), float(lm_info[8])), (float(lm_info[9]), float(lm_info[10]))
            fo = forient.FaceOrientation(leye, reye, nose_tip, lmouth, rmouth)
            fo_rect = fo.calc_orienation()

            if show_bbox_from_landmarks:
                self.draw_rect(img, fo.calc_rect(), scale=scale, color=(255, 20, 255))
                self.draw_rect(img, fo_rect, scale=scale, color=(255, 255, 20))

            self.draw_rect(img, fo.tight_crop(), scale=scale, color=(0, 155, 255))
            # self.draw_rect(img, fo.nik_crop(), scale=scale, color=(0, 255, 0))

            if show_rot_bbox:
                pts = fo.rot_crop()
                self.draw_poly(img, pts, scale=scale, color=(40, 40, 255))

            if show_landmarks:
                self.draw_landmarks(img, lm_info, scale=scale)

            info += f" ({img.shape[1]}x{img.shape[0]})"
            info_bottom = f"IP:{fo.in_plane: 4.1f}  LR:{fo.left_right_adjusted: 4.1f}  UD:{fo.up_down: 4.1f}"

        self.draw_info(img, info, info_bottom)
        return img


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("--bounding_box_file_name", type=str,
                        help=f"Input CelebA dataset bounding box file name.",
                        default="list_bbox_celeba.txt")
    parser.add_argument("--landmark_file_name", type=str,
                        help=f"Input CelebA dataset landmark file name.",
                        default="list_landmarks_celeba.txt")
    parser.add_argument("--images_dir", type=str,
                        help=f"Directory of images.",
                        default="images")
    args = parser.parse_args(argv)

    if not (mef.isfile(args.bounding_box_file_name) and
            mef.isfile(args.landmark_file_name) and
            mef.isdir(args.images_dir)):
        parser.print_help()
        sys.stdout.flush()
        time.sleep(0.2)
        raise ValueError("Bonding box and landmark must be files, and images dierctory must be a directory.")

    return args


def main(args):
    bbfile = open(args.bounding_box_file_name, 'r')
    lmfile = open(args.landmark_file_name, 'r')

    num_images = int(bbfile.readline())        # number of samples is same as number of bounding boxes
    num_lms = int(lmfile.readline())

    if num_images <= 0:
        print(f"ERROR: Number of bounding boxes ({num_images} must be > 0!")
        return -1

    if num_images != num_lms:
        print(f"ERROR: Numer of boxes ({num_images}) is different from number of landmark lines ({num_lms}.")
        return -1

    # Read headers.
    bbfile.readline()
    lmfile.readline()
    bb_lines = bbfile.readlines()
    lm_lines = lmfile.readlines()

    if not (len(bb_lines) == len(lm_lines) == num_images):
        print(f"ERROR: Inconsistent number of lines. Expected {num_images}.")
        return -1

    def line_to_list(line):
        line = line.strip('\n').strip('\r')
        line = re.sub('\\s+', ' ', line)  # remove duplicate spaces
        return line.split(' ')

    browser = CelebABrowser()
    browser.show_msg("CelebA Browser...", 3)
    idx = 0
    quit_ = False
    show_bbox = False
    show_landmarks = True
    show_bbox_from_landmarks = False
    show_rot_bbox = True

    while not quit_:
        bb_info, lm_info = line_to_list(bb_lines[idx]), line_to_list(lm_lines[idx])

        if bb_info[0] != lm_info[0]:
            img = browser.msg_img(f"ERROR: Line {idx+3}: filenames don't match!")
        else:
            fname = os.path.join(args.images_dir, bb_info[0])
            img = browser.load_and_render_image(fname, bb_info, lm_info, idx+1, num_images, show_bbox,
                                                show_landmarks, show_bbox_from_landmarks, show_rot_bbox)
        key = browser.show_img(img, -1)
        ckey = chr(key & 0xFF)
        lkey = ckey.lower()
        print(f"key is 0x{key:X}, char key is '{ckey}'.")

        if key in browser.RIGHT_ARROW_KEYS or ckey in ('n', ' '):
            # next pic (we go right)
            idx = min(num_images - 1, idx + 1)
        elif key in browser.LEFT_ARROW_KEYS or ckey in ('p',):
            # prev pic
            idx = max(0, idx - 1)
        elif ckey == 'j':
            def validate_jumpto(x):
                return x.isdigit() and 1 <= int(x) <= num_images

            resp = browser.prompt(f"Jump to (1..{num_images}):", validate_func=validate_jumpto)
            if resp is not None:
                idx = int(resp) - 1
        elif ckey == 'l':
            # toggle landmarks
            show_landmarks = not show_landmarks
        elif ckey == 'b':
            # toggle showing calculated bounding box based on landmarks
            show_bbox_from_landmarks = not show_bbox_from_landmarks
        elif ckey == 'm':
            show_bbox = not show_bbox
        elif ckey == 'r':
            show_rot_bbox = not show_rot_bbox
        elif ckey == 'h':
            browser.show_help(HELP_INFO)
        elif lkey == 'q' or key in (browser.ESCAPE_KEY, 0xFFFFFFFF):
            # quit
            quit_ = True


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
