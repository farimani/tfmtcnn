#######################################################################################
#
# mef.py
# Project General
#
# Mehran's handy helpers.
#
# Created by mehran on 11 / 06 / 18.
# Copyright © 2018 Percipo Inc. All rights reserved.
#
#
#######################################################################################

import os
import math
import time
import sys
from datetime import datetime
import numpy as np
from six import iteritems
from subprocess import Popen, PIPE
import imghdr
import struct
import hashlib


# noinspection PyUnusedLocal
def noop(*args, **kwargs):
    """ Ummmm.... """
    return None


def is_blank_string(s):
    """ Return true if s is None, empty or blank (all whitespace characters). """
    return s is None or s == '' or s.isspace()


def tsprint(*args, **kwargs):
    """ like print() but with a timestamp preceding each line... """
    print("["+str(datetime.now())+"] " + " ".join(map(str, args)), **kwargs)


def time_str():
    """ Return the current time in a string """
    return datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')


def time_me(func, iters=100):
    """
    Time a function...

    :param func:
    :param iters:
    :return:
    """
    start = time.time()
    for i in range(iters):
        func()
    return float(time.time() - start) / iters


def expand_seconds(seconds):
    """
    Return weeks, days, hours, mins and seconds from total seconds
    :param seconds:
    :return: weeks, days, hours, mins, seconds
    """
    weeks, seconds = divmod(seconds, 7 * 24 * 60 * 60)
    days, seconds = divmod(seconds, 24 * 60 * 60)
    hours, seconds = divmod(seconds, 60 * 60)
    minutes, seconds = divmod(seconds, 60)
    return int(weeks), int(days), int(hours), int(minutes), seconds


def expand_seconds_str(seconds):
    """
    Return a string denoting weeks, days, hours and seconds from expanding the seconds argument
    Second fractions are omitted.
    :param seconds:
    :return:
    """
    weeks, days, hours, minutes, seconds = expand_seconds(seconds)
    secs = int(seconds)

    if weeks > 0:
        ret = f"{weeks:2d}W, {days:2d}D, {hours:02d}h:{minutes:02d}m:{secs:02d}s"
    elif days > 0:
        ret = f"{days:2d}D, {hours:02d}h:{minutes:02d}m:{secs:02d}s"
    elif hours > 0:
        ret = f"{hours:02d}h:{minutes:02d}m:{secs:02d}s"
    elif minutes > 0:
        ret = f"{minutes:02d}m:{secs:02d}s"
    else:
        ret = f"{secs:02}s"

    return ret


def rect_size(r):
    """ Return w, h of the rectangle """
    return r[2]-r[0], r[3]-r[1]


def rect_width(r):
    """ Return width """
    return r[2]-r[0]


def rect_height(r):
    """ Return height """
    return r[3]-r[1]


def rect_area(r):
    """
    Return width * height.
    Note: Does not check for empty rectangles and will return wrong results.
    """
    return (r[2]-r[0]) * (r[3]-r[1])


def rect_is_empty(r):
    """ A rectangle is empty if it has width or height <= 0 """
    return r[0] >= r[2] or r[1] >= r[3]


def rect_make_empty():
    """ Return an empty rectangle """
    return [0, 0, -1, -1]


def rect_mid(r):
    """ Return the mid point of the rectangle """
    rw, rh = r[2] - r[0], r[3] - r[1]
    return r[0] + rw / 2, r[1] + rh / 2


def rect_set_mid(r, mid):
    """ Move the mid point of the rectangle to the given location, in place """
    rw, rh = rect_size(r)
    r[0] = (mid[0] - rw / 2)
    r[1] = (mid[1] - rh / 2)
    r[2] = r[0] + rw
    r[3] = r[1] + rh
    return r


def rect_at_mid(r, mid):
    """ Same as above, but not in place """
    return rect_set_mid(r.copy(), mid)


def rect_move(r, dx, dy):
    """ Move a rectangle by given delta, in place """
    r[0] += dx
    r[2] += dx
    r[1] += dy
    r[3] += dy
    return r


def rect_moved(r, dx, dy):
    """ Same as above, but not in place """
    return rect_move(r.copy(), dx, dy)


def rect_move_to(r, p):
    """ Move a rectangle to the given origin, in place """
    rect_move(r, p[0] - r[0], p[1] - r[1])
    return r


def rect_moved_to(r, p):
    """ Same as above, but not in place """
    return rect_move_to(r.copy(), p)


def rect_resize(r, dx, dy):
    """ Resize the rectangle by 2*dx, 2*dy, at the same center """
    r[0] += dx
    r[1] += dy
    r[2] -= dx
    r[3] -= dy
    return r


def rect_resized(r, dx, dy):
    """ Same as above, but not in place """
    return rect_resize(r.copy(), dx, dy)


def rect_flipped_vertical(r, width):
    """ Return a new flipped rectangle along the vertical axis of the image """
    # return [width - r[2] - 1, r[1], width - r[0] - 1, r[3]]
    return [width - r[2], r[1], width - r[0], r[3]]


def rect_contains_point(r, p):
    return r[0] <= p[0] <= r[2] and r[1] <= p[1] <= r[3]


def rect_intersection(r1, r2):
    """ Return the intersection of the two rects """
    return [max(r1[0], r2[0]), max(r1[1], r2[1]),
            min(r1[2], r2[2]), min(r1[3], r2[3])]


def rect_iou(r1, r2):
    """
    Intersection over union of two rects.

    There is a gross misnomer in machine vision lit.
    Union of two rectangles is the smallest rectangle that contains both rectangles. However, in machine vision,
    IOU, or intersection over union, is the area of the overlap over the *area* of both rectangles. i.e. it's
    (area1 + area2 - intersection_area).

    This routine, calculates the IOU according to the poorly named version in machine vision.

    :param r1:
    :param r2:
    :return:
    """
    r_intersect = rect_intersection(r1, r2)
    if rect_is_empty(r_intersect):    # no overlap
        return 0

    area_intersect = rect_area(r_intersect)
    return area_intersect / (rect_area(r1) + rect_area(r2) - area_intersect)


def dist_euclidean(v1, v2):
    """ Return euclidean distance b/w two vectors """
    return math.sqrt(np.sum(np.square(np.subtract(v1, v2))))


def dist_euclidean_sq(v1, v2):
    """ Return the square of euclidean distance b/w two vectors """
    return np.sum(np.square(np.subtract(v1, v2)))


def dist_cosine_similarity(v1, v2):
    """" Return cosine similiarity distance b/w two vectors """
    dot = np.sum(np.multiply(v1, v2))
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    similarity = dot / norm
    dist = np.arccos(similarity) / math.pi
    return dist


def to_rgb(frame):
    """ Make sure frame is 3 channels ... """
    if frame.ndim == 2:  # grayscale? make rgb
        h, w = frame.shape
        rgb = np.empty((h, w, 3), dtype=np.uint8)
        rgb[:, :, 0] = rgb[:, :, 1] = rgb[:, :, 2] = frame
    elif frame.shape[2] > 3:  # rgba? grab just rgb
        rgb = frame[:, :, 0:3]
    else:
        rgb = frame

    return rgb


def path_info(path, follow_links=True):
    """
    Check if a given path exists and return basic info. ~ is allowed.

    :param path:
    :param follow_links: follow sym links
    :return: type, size - where type is "file", "dir", "symlink", "other", or None if path doesn't exist.
    """
    path = os.path.expanduser(path)     # fix ~
    import stat

    if os.path.exists(path):
        st = os.stat(path) if follow_links else os.lstat(path)
        if stat.S_ISREG(st.st_mode):
            ret = "file"
        elif stat.S_ISDIR(st.st_mode):
            ret = "dir"
        elif stat.S_ISLNK(st.st_mode):
            ret = "symlink"
        else:
            ret = "other"

        return ret, st.st_size

    return None, 0


def path_type(path, follow_links=True):
    """
    Check if a given path exists and return its type. ~ is allowed.

    :param path:
    :param follow_links: follow sym links
    :return: type, where type is "file", "dir", "symlink", "other", or None if path doesn't exist.
    """
    ptype, size = path_info(path, follow_links)
    return ptype


def path_exists(path):
    """
    Return True if path exists...

    :param path:
    :return:
    """
    ptype, size = path_info(path)
    return ptype is not None


def isfile(path, follow_links=True):
    """
    Return True if the path points to a file.

    :param path:
    :param follow_links: follow sym links
    :return:
    """
    return path_type(path, follow_links=follow_links) == "file"


def isdir(path, follow_links=True):
    """
    Return True if the path points to a directory.

    :param path:
    :param follow_links: follow sym links
    :return:
    """
    return path_type(path, follow_links=follow_links) == "dir"


def issymlink(path):
    """
    Return True if the path points to a symbolic link.

    :param path:
    :return:
    """
    return path_type(path, follow_links=False) == "symlink"


def cwd():
    """
    :return: Current working directory
    """
    return os.getcwd()


def home_dir():
    """
    :return: Home directory
    """
    return os.path.expanduser("~")


def dirname(path):
    """
    Return the directory name part of a path. Returns "." if a dirname couldn't be parsed out.
    :param path:
    :return:
    """
    dname = os.path.dirname(path)
    return '.' if dname is '' else dname


def basename(path):
    """
    Return the basename part oa path.
    :param path:
    :return:
    """
    return os.path.basename(path)


def extension(path):
    """
    Return the extension, if one exists, incluing the "."
    :param path:
    :return:
    """

    _, ext = os.path.splitext(path)
    return ext


def create_dir_if_necessary(dirpath, recursive=True, raise_on_error=False):
    """
    Create a directory if it doesn't exist already. ~ is allowed.

    :param dirpath:
    :param recursive: Set to true to recrusively create all missing directories along the path.
    :param raise_on_error: If True, OSError is raised when error in making directory, and RunTimeError if path
           already exists but is not a directory. If False, error is returned in return code as below.
    :return: 0 if success, -1 if error while trying to create the directory, and -2 if path exists but is not
             a directory.
    """
    ptype, size = path_info(dirpath)
    ret = 0

    if recursive:
        path = dirpath
        dirs_to_create = []

        while True:
            if ptype is None:       # does not exist
                dirs_to_create.append(path)
            else:
                if ptype != "dir":
                    errmsg = f"ERROR: {path} already exists but is not a directory."
                    if raise_on_error:
                        raise RuntimeError(errmsg)

                    ret = -2
                    print(errmsg)
                    break

            path = dirname(path)
            if path in (".", '\\', '/'):
                break
            ptype, size = path_info(path)

        if ret == 0:
            # now create any paths that we need to along the way. traverse in reverse order
            for dirp in dirs_to_create[::-1]:
                try:
                    os.mkdir(dirp)
                except OSError as e:
                    if raise_on_error:
                        raise e

                    print(f"ERROR trying to make directory {dirp}.")
                    ret = -1
                    break
    else:
        if ptype is None:       # doesn't exist already
            try:
                os.mkdir(dirpath)
            except OSError as e:
                if raise_on_error:
                    raise e

                print(f"ERROR trying to make directory {dirpath}.")
                ret = -1
        elif ptype != "dir":
            errmsg = f"ERROR: {dirpath} already exists but is not a directory."
            if raise_on_error:
                raise RuntimeError(errmsg)

            print(errmsg)
            ret = -2

    return ret


def make_non_existent_pathname(path, create_as=None, raise_on_error=False):
    """
    Generated a file or directory name that doesn't exist already.
    If path+ext is available, it's returned, otherwise, an integer is inserted between
    path and ext as in "path.<int>.ext" where the path does not exist.

    If ext is empty, ".ext" is dropped from above.

    :param path: path for a file or a directory. If "" or None, just a numeric filename is generated.
    :param create_as: None: don't create, "file" create a zero sized file, "dir" create an empty dir
    :param raise_on_error: If True, OSError is raised when error in making directory, and RunTimeError if path
    :return: ret, path where ret is 0 if success, < 0 if creation fails.
    """
    path = "" if path is None else path
    path = os.path.expanduser(path)     # if it starts with ~
    path = os.path.realpath(path)
    path, ext = os.path.splitext(path)

    # Clean up path first. No trailing slash or "."
    cleanpath = path

    while len(cleanpath) > 0:
        if cleanpath[-1] in (os.sep, '.'):
            cleanpath = cleanpath[0:-1]
        else:
            break

    if cleanpath == "":
        cleanpath = "./0"

    pathname = cleanpath + ext
    i = 0

    while os.path.exists(pathname):
        pathname = cleanpath + "." + str(i) + ext
        i += 1

    ret = 0

    if create_as == "file":
        try:
            f = open(pathname, "w")
            f.close()
        except Exception as e:
            if raise_on_error:
                raise e

            print(f"Error creating empty file {pathname}.")
            print(f"{e}")
            ret = -1
    elif create_as == "dir":
        ret = create_dir_if_necessary(pathname, True, raise_on_error=raise_on_error)
    else:
        assert create_as is None

    return ret, pathname


def scan_dir(path, valid_exts=(), absolute=True, recursive=False):
    """
    Scan the given directory for files matching the given extentions

    :param path: Path to directory
    :param valid_exts: Must be a single string or a tuple of stringsd
    :param absolute: Convert paths to absolute
    :param recursive: Scan subdirectories
    :return: A list of file paths matching the given extension
    """
    fns = []
    valid_exts = "" if type(valid_exts) is tuple and len(valid_exts) == 0 else valid_exts

    for dirpath, dirnames, filenames in os.walk(path):
        for fn in (x for x in filenames if x.endswith(valid_exts)):
            fn = os.path.abspath(os.path.join(dirpath, fn)) if absolute else os.path.join(dirpath, fn)
            fns.append(fn)
        if not recursive:
            break

    return fns


def get_line_count(fname):
    """
    Get the total line count in a given file.
    :param fname:
    :return:
    """

    # return sum(1 for line in open(fname))
    with open(fname, 'rb') as f:
        read_func = f.read   # silly optimization!
        count = 0
        while True:
            buff = read_func(1024*1024)
            if not buff:
                break
            count += buff.count(b'\n')

        return count


def read_non_blank_line(f, comment_char=None):
    """
    Returns number of lines read and the next non-blank line (cnt, line)
    Return an (cnt, None) if end of file reached.

    comment_char can specify a character that denotes a comment line, if it starts
    with that character as its first nonblank character.
    """
    cnt = 0

    while True:
        line = f.readline()
        if not line:        # eof
            break

        cnt += 1
        ls = line.strip()

        if ls and ls[0] is not comment_char:
            break

    return cnt, line


def read_non_blank_lines(f, start_line_num=0, comment_char=None):
    """
    Returns an array of line numbers and an array of corresponding non-blank lines.

    comment_char can specify a character that denotes a comment line, if it starts
    with that character as its first nonblank character.
    """
    line_num = start_line_num
    line_nums = []
    lines = []

    while True:
        cnt, line = read_non_blank_line(f, comment_char)
        if not line:        # eof
            break

        line_num += cnt
        line_nums.append(line_num)
        lines.append(line)

    return line_nums, lines


def gaussian(x, mu, sig):
    """
    A Gaussian distribution

    :param x:
    :param mu:
    :param sig:
    :return:
    """
    return np.exp(-((x - mu)**2) / (2 * sig**2))


def standardize_image_variance_adjust(x):
    """
    Standardize image according to its variance. Scale image pixels to zero mean and unit variance.
    Same as tf.image.per_image_standardization op

    Computes `(x - mean) / adjusted_stddev`, where `mean` is the average of all values in image, and
    `adjusted_stddev = max(stddev, 1.0/sqrt(image.NumElements()))`.

    `stddev` is the standard deviation of all values in `image`. It is capped
    away from zero to protect against division by 0 when handling uniform images.

    """
    std = max(x.std(), 1/math.sqrt(x.size))
    return np.multiply(np.subtract(x, x.mean()), 1/std)


def standardize_image_fixed(x):
    """
    Standardize image so that its pixel values go from -1 to 1. Scale image pixels to -1 to 1.
    :param x:
    :return:
    """
    return (x - 127.5) / 128


def print_arguments(args, filename=None, print_header=True):
    """
    Print the arguments dictionary to a file, or stdout if filename is blank or None
    Copied from dave sandberg's facenet.

    :param args:
    :param filename:
    :param print_header: Print a header line
    :return:
    """

    f = sys.stdout if is_blank_string(filename) else open(filename, 'w')

    if print_header:
        f.write(f"Program arguments for {basename(sys.argv[0])}:\n\n")

    for key, value in iteritems(vars(args)):
        f.write(f"{key}: {value}\n")


def log_arguments(args, log_dir=None):
    """ Ptint the argumemnts dictionary to program_name.TIMESTR.args.txt and to stdout """

    log_dir = "." if log_dir is None else os.path.expanduser(log_dir)
    program_name = basename(sys.argv[0])
    log_filename = os.path.join(log_dir, program_name + "." + time_str() + ".args.txt")
    print_arguments(args, log_filename)
    print(f"Program aguments for {basename(sys.argv[0])} (written to {log_filename}):")
    print_arguments(args)


def store_git_revision_info(src_path, output_dir, tf_version=None):
    """
    Store the git revision info the given source directory, as well as program arguments and tensorflow version.
    Copied from dave sandberg's facenet.

    :param src_path:
    :param output_dir:
    :param tf_version: tensorflow version - optional
    :return:
    """
    cmd = ['git', 'rev-parse', 'HEAD']

    try:
        # Get git hash
        gitproc = Popen(cmd, stdout=PIPE, cwd=src_path)
        (stdout, _) = gitproc.communicate()
        git_hash = stdout.strip()
    except OSError as e:
        git_hash = ' '.join(cmd) + ': ' + e.strerror

    try:
        # Get local changes
        cmd = ['git', 'diff', 'HEAD']
        gitproc = Popen(cmd, stdout=PIPE, cwd=src_path)
        (stdout, _) = gitproc.communicate()
        git_diff = stdout.strip()
    except OSError as e:
        git_diff = ' '.join(cmd) + ': ' + e.strerror

    # Store a text file in the log directory
    rev_info_filename = os.path.join(output_dir, 'revision_info.txt')
    with open(rev_info_filename, "w") as text_file:
        arg_string = ' '.join(sys.argv)
        text_file.write(f"arguments: {arg_string}\n--------------------\n")

        if tf_version is not None:
            text_file.write(f"tensorflow version: {tf_version}\n--------------------\n")

        text_file.write(f"git hash: {git_hash}\n--------------------\n")
        text_file.write(f"{git_diff}")


def write_network_arch(filename, end_points):
    """
    Write out the network architecture to a text file, from the end_points dictionary.

    :param filename:
    :param end_points:
    :return:
    """
    hd = open(filename, 'w')

    for key in end_points.keys():
        info = f"{key}:{end_points[key].get_shape().as_list()}\n"
        hd.write(info)
    hd.close()


def execute(cmd):
    """
    Execute a subprocess and display its output line by line
    """

    with Popen(cmd, stdout=PIPE, bufsize=1, universal_newlines=True) as p:
        for line in p.stdout:
            print(line, end='')  # process line here

        return p.returncode


def get_image_size(fname):

    """
    Determine the image type of fhandle and return its size.
    from draco

    return width,height or -1, -1
    """

    with open(fname, 'rb') as fhandle:
        head = fhandle.read(24)
        if len(head) != 24:
            return -1, -1

        if imghdr.what(fname) == 'png':
            check = struct.unpack('>i', head[4:8])[0]
            if check != 0x0d0a1a0a:
                return -1, -1

            width, height = struct.unpack('>ii', head[16:24])
        elif imghdr.what(fname) == 'gif':
            width, height = struct.unpack('<HH', head[6:10])
        elif imghdr.what(fname) == 'jpeg':
            # noinspection PyBroadException,PyBroadException
            try:
                fhandle.seek(0)  # Read 0xff next
                size = 2
                ftype = 0
                while not 0xc0 <= ftype <= 0xcf:
                    fhandle.seek(size, 1)
                    byte = fhandle.read(1)

                    while ord(byte) == 0xff:
                        byte = fhandle.read(1)

                    ftype = ord(byte)
                    size = struct.unpack('>H', fhandle.read(2))[0] - 2

                # We are at a SOFn block
                fhandle.seek(1, 1)  # Skip `precision' byte.
                height, width = struct.unpack('>HH', fhandle.read(4))
            except Exception:  # IGNORE:W0703
                return -1, -1
        else:
            return -1, -1

        return width, height


def md5sum(fname):
    """
    Calculate the md5 sum of a file

    :param fname:
    :return:
    """

    hash_md5 = hashlib.md5()

    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()


class ProgressText:
    def __init__(self, total, current=0, newline_when_done=True):
        self._total = 0
        self._start = 0
        self._current = 0
        self._newline_when_done = newline_when_done
        self._start_time = 0
        self._last_update = 0
        # self._elapsed_times = deque(maxlen=30)
        self._update_time_acc = AccBlend()
        self.reset(total=total, current=current, newline_when_done=newline_when_done)

    def reset(self, total=None, current=None, newline_when_done=None):
        current = self._start if current is None else current
        total = self._total if total is None else total
        self._total = max(total, current)
        self._start = self._current = current
        self._newline_when_done = newline_when_done
        self._start_time = self._last_update = time.time()
        # self._elapsed_times.clear()
        self._update_time_acc.reset()

    @classmethod
    def clean(cls):
        """Clean the line with blanks ..."""
        print("\r", end='')     # MEF: Python 3 seems to clear the line when you do a \r. No need for printing spaces...

    def show(self, msg="", clean=False):
        """ Show without updating counters, etc. """
        if clean:   # clean line first
            self.clean()

        percent_done = self.percent_done()
        if msg is None or msg == "":
            print(f"\r{percent_done:-5.1f}% - {self.time_remaining()}", end='', flush=True)
        else:
            print(f"\r{msg} ({percent_done:-5.1f}% - {self.time_remaining()})", end='', flush=True)

    def update_current_time(self, msg=""):
        """
        Only update the current elapsed time and show the msg with new remaining time estimtate
        :return:
        """
        elapsed = time.time() - self._last_update
        self._update_time_acc.blend(elapsed, 0.9999)
        self.show(msg)

    def time_remaining(self):
        # avg_update_time = sum(self._elapsed_times) / len(self._elapsed_times)
        avg_update_time = self._update_time_acc.value()
        return expand_seconds_str((self._total - self._current) * avg_update_time)

    def percent_done(self):
        return self._current * 100.0 / self._total

    def update(self, msg="", step=1):
        if self._current >= self._total:
            return

        self._current = min(self._current + step, self._total)
        now = time.time()
        elapsed = now - self._last_update
        self._last_update = now
        # self._elapsed_times.append(elapsed / step)
        self._update_time_acc.blend(elapsed / step, 0.9999)
        self.show(msg)

        if self._current == self._total:
            if self._newline_when_done:
                print("")
            # else:
            #     print("\r                                                   ")


class AccBlend:
    """
    Class to blend values of a series using an accumulator.

    Use it like this:
        acc = AccBlend()
        acc.blend(35, 0.7)
        acc.blend(90, 0.7)
        print(acc.value())
    """

    def __init__(self):
        self._acc = self._count_acc = None
        self.reset()

    def blend(self, amount, decay, weight=1.0):
        self._acc = self._acc * decay + (amount * weight)
        self._count_acc = self._count_acc * decay + weight
        return self

    def init(self, amount, weight=1.0):
        self.blend(amount, 0, weight)
        return self

    def reset(self):
        self._acc = self._count_acc = 0.0
        return self

    def is_empty(self):
        return self._count_acc == 0.0

    def value(self):
        return self._acc / (1.0 if self._count_acc == 0 else self._count_acc)

    # noinspection PyProtectedMember
    def __iadd__(self, other):
        self._acc += other._acc
        self._count_acc += other._count_acc
        return self


