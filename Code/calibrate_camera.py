#!/usr/env/bin python3

"""
CMSC733 Spring 2022: Classical and Deep Learning Approaches for Geometric Computer Vision
Homework 1: AutoCalib

Author(s):
Tanuj Thakkar (tanuj@umd.edu)
M. Engg Robotics
University of Maryland, College Park
"""

import sys
import cv2
import os
import numpy as np
import time
import argparse

from utils import *

sys.dont_write_bytecode = True


class CalibrateCamera:

    def __init__(self, data_dir: str) -> None:
        self.img_set = create_image_set(data_dir)
        self.homography_set = list()

    def __estimate_homography_set(self, box_size: float, num_pts_x: int, num_pts_y: int) -> np.array:

        for img in self.img_set:
            img = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(img_gray, (num_pts_x, num_pts_y), None)


            X, Y = np.meshgrid(np.linspace(0, num_pts_x - 1, num_pts_x), np.linspace(0, num_pts_y - 1, num_pts_y))
            X = np.flip((X.reshape(54, 1) * box_size), axis=0)
            Y = (Y.reshape(54, 1) * box_size)
            M = np.float32(np.hstack((Y, X)))

            if ret:
                corners = corners.reshape(-1, 2)
                corners = cv2.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))

                for corner in corners:
                    cv2.circle(img, (int(corner[0]), int(corner[1])), 2, (0,255,0), 2)

                H = cv2.findHomography(M, corners)[0]
                self.homography_set.append(H)

        self.homography_set = np.array(self.homography_set)
        return self.homography_set

    def estimate_intrinsics(self):

        self.__estimate_homography_set(21.5, 9, 6)