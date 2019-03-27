#######################################################################################
#
# wider_browser.py
# Project MTCNN
#
# GUI browser for WIDER Face dataset
#
# Created by mehran on 03 / 25 / 19.
# Copyright © 2019 Percipo Inc. All rights reserved.
#
#######################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import argparse

import cv2

import mef
import browser_gui

from tfmtcnn.datasets.WIDERFaceDataset import WIDERFaceDataset

WIN_TITLE = "WIDER Face Browser"

HELP_INFO = [("'h'",                        "This help screen"),
             (),
             ("Arrows, 'p', 'n', ' '",      "Navigate"),
             ("'j'",                        "Jump to an image"),
             (),
             ("'l'",                        "Toggle landmarks"),
             ("'b'",                        "Toggle extra bounding boxes"),
             ("'r'",                        "Toggle rotated cropped box"),
             (),
             ("ESC or 'q'",                 "Quit")]

class WIDERFaceBrowser(browser_gui.BrowserGUI):
    def __init__(self):
        super().__init__(WIN_TITLE)

    def draw_bboxes(self, img, bboxes, scale=1.0, color=(20, 255, 20)):
        for bbox in bboxes:
            bb = [int(x * scale) for x in bbox]
            self.draw_rect(img, bb, color=color)

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

    def load_and_render_image(self, fname, bboxes, image_num, total_images):
        info = f"{image_num} / {total_images} - {mef.basename(fname)}"
        info_bottom = ""
        img, scale = self.imread_and_scale(fname)

        if img is None:
            img = self.msg_img(f"ERROR: Can't read image {mef.basename(fname)}")
            info += " INVALID"
        else:
            self.draw_bboxes(img, bboxes, scale=scale)
            info += f" ({img.shape[1]}x{img.shape[0]})"

        self.draw_info(img, info, info_bottom)
        return img


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("--bounding_box_file_name", type=str,
                        help=f"Input WIDER Face dataset bounding box file name.",
                        default="wider_face_train_bbx_gt.txt")
    parser.add_argument("--images_dir", type=str,
                        help=f"Directory of images.",
                        default="images")
    args = parser.parse_args(argv)

    if not (mef.isfile(args.bounding_box_file_name) and
            mef.isdir(args.images_dir)):
        parser.print_help()
        sys.stdout.flush()
        time.sleep(0.2)
        raise ValueError("Bonding box must be a file, and images dierctory must be a directory.")

    return args


def main(args):
    wfd = WIDERFaceDataset()
    if not wfd.read(args.images_dir, args.bounding_box_file_name, min_size=0):
        print(f"ERROR reading dataset from {args.bounding_box_file_name}, images dir: {args.images_dir}.")
        return -1

    dataset = wfd.data()
    image_fns, bboxes, total_faces = dataset['images'], dataset['bboxes'], dataset['number_of_faces']
    num_images = len(image_fns)

    browser = WIDERFaceBrowser()
    browser.show_msg("WIDER Face Browser...", 3)
    idx = 0
    quit_ = False

    while not quit_:
        img_bboxes = bboxes[idx]
        img = browser.load_and_render_image(image_fns[idx], img_bboxes, idx+1, num_images)
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
        elif ckey == 'h':
            browser.show_help(HELP_INFO)
        elif lkey == 'q' or key in (browser.ESCAPE_KEY, 0xFFFFFFFF):
            # quit
            quit_ = True
        elif ckey == 'j':
            def validate_jumpto(x):
                return x.isdigit() and 1 <= int(x) <= num_images

            resp = browser.prompt(f"Jump to (1..{num_images}):", validate_func=validate_jumpto)

            if resp is not None:
                idx = int(resp) - 1


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
