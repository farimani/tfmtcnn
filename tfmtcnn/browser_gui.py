#######################################################################################
#
# browser_gui.py
# Project General
#
# GUI primitives for a simple face dataset browser.
#
# Created by mehran on 03 / 25 / 19.
# Copyright © 2019 Percipo Inc. All rights reserved.
#
#######################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import cv2


class BrowserGUI:
    RIGHT_ARROW_KEYS = (0x270000,  # windows
                        )

    LEFT_ARROW_KEYS = (0x250000,  # windows
                       )

    ESCAPE_KEY = 0x1B
    BACKSPACE_KEY = 0x08
    TAB_KEY = 0x09
    ENTER_KEY = 0x0D

    def __init__(self, win_title):
        self._win_title = win_title
        cv2.namedWindow(win_title, cv2.WINDOW_GUI_EXPANDED)

        self._msg_win_size = (640, 480)
        self._color = (255, 255, 255)
        self._msg_color = (0, 60, 255)
        self._fface = cv2.FONT_HERSHEY_SIMPLEX
        self._fscale = 1.5
        self._fthick = 2
        self._moved_window = False

    def _line_info(self, txt):
        """
        Split the text into multiple lines and return the max width, height and baseline
        :param txt:
        :return:
        """
        lines = txt.splitlines()
        max_tw, th, baseline = -1, 0, 0

        for line in lines:
            tw, th, baseline = self.get_text_size(line)

            if tw > max_tw:
                max_tw = tw

        return lines, max_tw, th, baseline

    def set_msg_win_size(self, width, height):
        self._msg_win_size = (width, height)
        return self

    def set_color(self, color):
        self._color = color

    def get_color(self):
        return self._color

    def set_msg_color(self, color):
        self._msg_color = color

    def get_msg_color(self):
        return self._msg_color

    def get_font(self):
        return self._fface, self._fscale, self._fthick

    def set_font(self, face=None, scale=None, thick=None):
        self._fface = face if face is not None else self._fface
        self._fscale = scale if scale is not None else self._fscale
        self._fthick = thick if thick is not None else self._fthick

    def draw_line(self, img, x1, y1, x2, y2, thick=1, color=None):
        color = self._color if color is None else color
        cv2.line(img, (x1, y1), (x2, y2), color, thick, lineType=cv2.LINE_AA)

    def draw_rect(self, img, rect, scale=1.0, thick=1, color=None):
        color = self._color if color is None else color
        r = [int(round(x * scale)) for x in rect]
        cv2.rectangle(img, (r[0], r[1]), (r[2], r[3]), color, thick)

    def draw_poly(self, img, points, scale=1.0, thick=1, color=None):
        color = self._color if color is None else color
        pts = [(int(round(x * scale)), int(round(y * scale))) for x, y in points]
        for idx in range(1, len(pts)):
            self.draw_line(img, pts[idx-1][0], pts[idx-1][1], pts[idx][0], pts[idx][1], color=color, thick=thick)

        self.draw_line(img, pts[-1][0], pts[-1][1], pts[0][0], pts[0][1], color=color, thick=thick)

    def draw_circle(self, img, cx, cy, radius, thick=1, color=None):
        color = self._color if color is None else color
        cv2.circle(img, (cx, cy), radius, color=color, thickness=thick, lineType=cv2.LINE_AA)

    def put_text(self, img, text, x, y, fface=None, fscale=None, color=None, fthick=None):
        fface = self._fface if fface is None else fface
        fscale = self._fscale if fscale is None else fscale
        fthick = self._fthick if fthick is None else fthick
        color = self._color if color is None else color
        cv2.putText(img, text, (x, y), fface, fscale, color, fthick, lineType=cv2.LINE_AA)

    def get_text_size(self, text, fface=None, fscale=None, fthick=None):
        fface = self._fface if fface is None else fface
        fscale = self._fscale if fscale is None else fscale
        fthick = self._fthick if fthick is None else fthick
        (tw, th), baseline = cv2.getTextSize(text, fface, fscale, fthick)
        return tw, th, baseline

    def msg_img_ex(self, msg, color=None, halign="center", valign="center"):
        """
        Create an image with a one liner msg txt.

        :param msg:
        :param color:
        :param halign: center, left, right
        :param valign: center, top, bottom
        :return: image, tx, ty, tw, th, baseline of the last line
        """
        color = self._msg_color if color is None else color
        imh, imw = self._msg_win_size
        hmargin, vmargin, line_spacing = 5, 10, 3
        img_msg = np.zeros((imh, imw, 3), dtype=np.uint8)

        lines, tw, th, baseline = self._line_info(msg)
        num_lines = len(lines)
        fscale = self._fscale

        if tw > imw - hmargin * 2:
            fscale *= (imw - hmargin * 2) / tw

        total_height = (th * num_lines) + (line_spacing * (num_lines - 1))
        tx = 0

        if valign == "center":
            ty = (imh - total_height) // 2 - baseline
        elif valign == "top":
            ty = th + vmargin + baseline
        else:  # bottom
            ty = imh - vmargin - total_height - baseline

        for i, line in enumerate(lines):
            tw, th, baseline = self.get_text_size(line, fscale=fscale)

            if halign == "center":
                tx = (imw - tw) // 2
            elif halign == "left":
                tx = hmargin
            else:  # right
                tx = imw - hmargin - tw

            self.put_text(img_msg, line, tx, ty, fscale=fscale, color=color)

            if i < num_lines - 1:
                ty += th + line_spacing

        return img_msg, tx, ty, tw, th, baseline

    def msg_img(self, msg, color=None, halign="center", valign="center"):
        """
        Same as msg_img_ex but returns just the image and no text attributes

        :param msg:
        :param color:
        :param halign:
        :param valign:
        :return:
        """
        ret = self.msg_img_ex(msg, color, halign, valign)
        return ret[0]

    def show_img(self, img, delay=None):
        cv2.imshow(self._win_title, img)

        if not self._moved_window:
            cv2.moveWindow(self._win_title, 10, 30)
            self._moved_window = True

        cv2.resizeWindow(self._win_title, img.shape[1], img.shape[0])
        if delay is not None:
            return cv2.waitKeyEx(delay) & 0xFFFFFFFF

        return 0xFFFFFFFF

    def show_msg(self, msg, delay=None, color=None, halign="center", valign="center"):
        image = self.msg_img(msg, color=color, halign=halign, valign=valign)
        return self.show_img(image, delay)

    def prompt(self, msg, default_input="", validate_func=None):
        done = False
        input_so_far = default_input
        input_color = (200, 200, 255)
        cursor_color = (128, 128, 128)
        underline_color = (40, 40, 40)
        error_flash = False

        while not done:
            mimg, tx, ty, tw, th, baseline = self.msg_img_ex(msg)
            imh, imw, _ = mimg.shape
            input_tw, input_th, baseline = self.get_text_size(input_so_far)
            tx = (imw - input_tw) // 2
            ty += input_th + th - baseline

            inp_color = input_color if not error_flash else (0, 160, 255)
            und_color = underline_color if not error_flash else (0, 160, 255)
            self.draw_line(mimg, tx-10, ty, tx+input_th+10, ty, color=und_color)
            self.put_text(mimg, input_so_far, tx, ty, color=inp_color)
            cursor_x, cursor_y = (tx + input_tw + 3, ty - input_th - 2)
            self.draw_line(mimg, cursor_x, cursor_y, cursor_x, cursor_y+th+4, color=cursor_color)

            if error_flash:
                self.show_img(mimg, 1)
                time.sleep(0.1)
                input_so_far = input_so_far[:-1]
                error_flash = False
            else:
                key = self.show_img(mimg, -1)
                ckey = chr(key & 0xFF)
                print(f"key is 0x{key:X}, char key is '{ckey}'.")

                if key == self.ESCAPE_KEY:
                    input_so_far = None
                    done = True
                elif key == self.ENTER_KEY:
                    done = True
                elif key == self.BACKSPACE_KEY:
                    input_so_far = input_so_far[:-1]
                elif ckey.isalnum():
                    input_so_far += ckey
                    error_flash = validate_func is not None and not validate_func(input_so_far)

        return input_so_far

    def show_help(self, help_info):
        """
        Help info is a list of pairs, first component is the key/action and second is description.
        :param help_info:
        :return:
        """
        font_saved = self.get_font()
        col_saved = self.get_msg_color()
        self.set_font(scale=0.8)
        self.set_msg_color((120, 120, 255))
        mimg, tx, ty, tw, th, baseline = self.msg_img_ex(self._win_title + " Help", valign="top")
        fscale, fthick = 0.4, 1
        imh, imw, _ = mimg.shape
        ty += 2 * th
        _, th, _ = self.get_text_size("|", fscale=fscale, fthick=fthick)
        col1, col2 = (180, 180, 255), (180, 255, 255)

        for hinfo in help_info:
            if len(hinfo) > 0:
                self.put_text(mimg, hinfo[0] + ":", 5, ty, fscale=fscale, fthick=fthick, color=col1)

                if len(hinfo) > 1:
                    self.put_text(mimg, hinfo[1], 150, ty, fscale=fscale, fthick=fthick, color=col2)

            self.draw_line(mimg, 5, ty+5, imw - 5, ty+5, color=(60, 60, 60))
            ty += th + 7

        self.set_font(*font_saved)
        self.set_msg_color(col_saved)
        self.show_img(mimg, delay=-1)

    @staticmethod
    def imread(fname):
        return cv2.imread(fname)

    @classmethod
    def imread_and_scale(cls, fname, min_width=0, min_height=0, max_width=0, max_height=0):
        min_width = 800 if min_width <= 0 else min_width
        min_height = 600 if min_height <= 0 else min_height
        max_width = 1024 if max_width <= 0 else max_width
        max_height = 700 if max_height <= 0 else max_height

        img = cls.imread(fname)
        scale = None

        if img is not None:
            scale = 1.0
            im_height, im_width, _ = img.shape

            if im_width > max_width or im_height > max_height:
                if im_width / max_width > im_height / max_height:
                    scale = max_width / im_width
                else:
                    scale = max_height / im_height
            elif im_width < min_width or im_height < min_height:
                if min_width / im_width > min_height / im_height:
                    scale = min_width / im_width

                    if scale * im_height > max_height:
                        scale = max_height / im_height
                else:
                    scale = min_height / im_height

                    if scale * im_width > max_width:
                        scale = max_width / im_width

            if scale != 1.0:
                nw, nh = int(round(scale * im_width)), int(round(scale * im_height))
                img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

        return img, scale
