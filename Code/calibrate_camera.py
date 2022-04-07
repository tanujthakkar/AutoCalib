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
        self.K = None

    def __estimate_homography_set(self, box_size: float, num_pts_x: int, num_pts_y: int) -> np.array:

        for img in self.img_set:
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

    def estimate_intrinsics(self) -> np.array:

        self.__estimate_homography_set(21.5, 9, 6)

        def v_ij(homography: np.array, i: int, j: int) -> np.array:
            return np.array([homography[0,i] * homography[0,j],
                             homography[0,i] * homography[1,j] + homography[1,i] * homography[0,j],
                             homography[1,i] * homography[1,j],
                             homography[2,i] * homography[0,j] + homography[0,i] * homography[2,j],
                             homography[2,i] * homography[1,j] + homography[1,i] * homography[2,j],
                             homography[2,i] * homography[2,j]]).reshape(-1, 1)

        V = np.empty([0,6])
        for i, homography in enumerate(self.homography_set):
            V = np.append(V, v_ij(homography, 0, 1).transpose(), axis=0)
            V = np.append(V, np.subtract(v_ij(homography, 0, 0), v_ij(homography, 1, 1)).transpose(), axis=0)

        U, S, V_t = np.linalg.svd(V)
        b_11, b_12, b_22, b_13, b_23, b_33 = V_t[-1]

        v_0 = (b_12*b_13 - b_11*b_23)/(b_11*b_22 - b_12**2)
        lambda_ = b_33 - ((b_13**2 + (v_0*(b_12*b_13 - b_11*b_23)))/b_11)
        alpha = np.sqrt(lambda_/b_11)
        beta = np.sqrt((lambda_*b_11)/(b_11*b_22 - b_12**2))
        gamma = (-b_12*(alpha**2)*beta)/lambda_
        u_0 = (gamma*v_0/beta) - (b_13*(alpha**2)/lambda_)

        K = np.array([[alpha, gamma, u_0],
                      [0, beta, v_0],
                      [0, 0, 1]])

        self.K = K
        return K