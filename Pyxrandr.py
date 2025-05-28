import numpy as np
import subprocess
import cv2

class Pyxrandr:
    """Interfacing with the process xrandr by calculating transformation matrix etc.

    Heavily inspired by https://github.com/qurn/xrandr-keystone-helper.

    :param dimension: dimensions of the output (beamer) like (1920, 1080)
    :type dimension: tuple<int>
    :param mon: name of the connection to the monitor/beamer (like 'eDP1')
    :type mon: str
    """
    def __init__(self, dimension, mon):
        self.mon = mon
        self.w, self.h = tuple(dimension)

    def transform(self, pos):
        """Calculate the soll-position of the registered tags (see camera modules Camera.py) based on the ist-position, calculate the matrix for xrandr and call.
        
        :param pos: positions of the detected corners of the projected apriltags in a relative camera coordinate system (0-1)
        :type pos: dict<tuple<float>>
        """
        # 1. get cameramtx from cv2 for the camera/table corners to the tag corners
        # 2. invert mtx * scaling