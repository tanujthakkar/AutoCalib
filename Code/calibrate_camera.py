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
from scipy.optimize import least_squares

from utils import *

sys.dont_write_bytecode = True


class CalibrateCamera:

    def __init__(self, data_dir: str, result_dir: str, visualize: bool = False) -> None:
        self.img_set = create_image_set(data_dir)
        self.rect_img_set = None
        self.M = None
        self.m = None
        self.homography_set = list()
        self.K = None
        self.Rt = None
        self.params = None
        self.result_dir = result_dir
        self.visualize = visualize

    def __estimate_homography_set(self, box_size: float, num_pts_x: int, num_pts_y: int) -> np.array:

        m = np.empty([0,54,3])
        for i, img in enumerate(self.img_set):
            # img = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(img_gray, (num_pts_x, num_pts_y), None)

            X, Y = np.meshgrid(np.linspace(0, num_pts_x - 1, num_pts_x), np.linspace(0, num_pts_y - 1, num_pts_y))
            # X = (X.reshape(54, 1) * box_size)
            X = np.flip((X.reshape(54, 1) * box_size), axis=0)
            Y = (Y.reshape(54, 1) * box_size)
            M = np.float32(np.hstack((Y, X)))
            M = np.hstack((M, np.ones([1, len(M)]).transpose()))
            self.M = M

            if ret:
                corners = corners.reshape(-1, 2)
                corners = cv2.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
                corners = np.hstack((corners, np.ones([1, len(corners)]).transpose()))
                m = np.insert(m, len(m), corners, axis=0)

                img_ = np.copy(img)
                # img_ = img
                for corner in corners:
                    cv2.circle(img_, (int(corner[0]), int(corner[1])), 4, (0,255,0), 4)

                cv2.imwrite(os.path.join(self.result_dir, str(i) + '.png'), img_)
                # cv2.imshow("", img)
                # cv2.waitKey()
                H = cv2.findHomography(M, corners)[0]
                self.homography_set.append(H)

        self.m = m
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
        gamma = -b_12*((alpha**2)*beta)/lambda_
        u_0 = (gamma*v_0/beta) - (b_13*(alpha**2)/lambda_)

        K = np.array([[alpha, gamma, u_0],
                      [0, beta, v_0],
                      [0, 0, 1]])

        # self.K = K
        # print("K:\n", K)
        return K

    def estimate_extrinsics_set(self, K:np.array) -> np.array:

        def estimate_extrinsics(homography: np.array, K: np.array) -> np.array:

            K_inv = np.linalg.inv(K)
            lambda_ = 1/np.linalg.norm(np.dot(K_inv, homography[:,0]), ord=2)

            # Rt = np.zeros([3,4])

            R = np.zeros([3,3])
            R[:,0] = lambda_ * np.dot(K_inv, homography[:,0])
            R[:,1] = lambda_ * np.dot(K_inv, homography[:,1])
            R[:,2] = np.cross(R[:,0], R[:,1])
            t = (lambda_ * np.dot(K_inv, homography[:,2])).reshape(3,1)

            Q = np.array([R[:,0], R[:,1], R[:,2]]).transpose()
            U, S, Vt = np.linalg.svd(Q)
            R = np.dot(U, Vt)

            Rt = np.hstack([R, t])

            return Rt

        Rt = np.empty([0,3,4])
        for homography in self.homography_set:
            Rt = np.insert(Rt, len(Rt), estimate_extrinsics(homography, K), axis=0)

        # self.Rt = Rt
        return Rt

    def projection_error(self, m: np.array, M: np.array, K: list, Rt: np.array, k1: float, k2: float, img: np.array = None) -> float:
        K = np.array([[K[0], K[1], K[2]],
                      [0, K[3], K[4]],
                      [0, 0, 1]])

        m = m.reshape(3,1)

        M_3D = np.array([M[0], M[1], 0, 1]).reshape(4, 1)
        M_ = np.dot(Rt, M_3D)
        M_ = M_/M_[-1]

        distortion_radius = (M_[0]**2 + M_[1]**2)[0]

        m_ = np.dot(K, M_)
        m_ = m_/m_[-1]
        u, v, _ = m_
        u_0, v_0 = K[0,2], K[1,2]
        u_ = u + (u - u_0) * (k1*distortion_radius + k2*(distortion_radius**2))
        v_ = v + (v - v_0) * (k1*distortion_radius + k2*(distortion_radius**2))
        if(img is not None):
            cv2.circle(img, (int(u_[0]), int(v_[0])), 4, (0,0,255), 4)
        m_ = np.array([u_[0], v_[0], 1]).reshape(3,1)
        
        return np.linalg.norm(np.subtract(m, m_), ord=2)

    def projection_loss(self, params: list, M: np.array, Rt: np.array, img: np.array = None) -> float:
        loss = 0
        for i, corners in enumerate(self.m):
            for j, corner in enumerate(corners):
                loss += self.projection_error(corner, M[j], params[:5], Rt[i], params[-2], params[-1], img)

        return loss

    def optimize_params(self) -> np.array:
        params = least_squares(fun=self.projection_loss, x0=[self.K[0,0], self.K[0,1], self.K[0,2], self.K[1,1], self.K[1,2], 0, 0], args=[self.M, self.Rt, None])

        self.params = params['x']

        rect_K = np.array([[self.params[0], self.params[1], self.params[2]],
                      [0, self.params[3], self.params[4]],
                      [0, 0, 1]])
        kc = np.array([self.params[5], self.params[6], 0., 0.])

        return rect_K, kc

    def reprojection_img_loss(self, params: list, m:np.array, M: np.array, Rt: np.array, img: np.array = None) -> float:
        loss = 0
        for j, corner in enumerate(m):
            loss += self.projection_error(corner, M[j], params[:5], Rt, params[-2], params[-1], img)

        return loss

    def reprojection_error(self, rect_K: np.array, kc: np.array) -> float:
        # rect_K = np.array([[self.params[0], self.params[1], self.params[2]],
        #               [0, self.params[3], self.params[4]],
        #               [0, 0, 1]])
        # kc = np.array([self.params[5], self.params[6], 0., 0.])
        
        # rect_K = self.K
        # kc = np.array([2.26937511e-02, -1.42563484e-01, 0., 0.])

        rect_Rt = self.estimate_extrinsics_set(rect_K)

        reprojection_loss = 0
        for i, img in enumerate(self.img_set):
            rect_img = cv2.undistort(img, rect_K, kc)
            params = [rect_K[0,0], rect_K[0,1], rect_K[0,2], rect_K[1,1], rect_K[1,2], kc[0], kc[1]]
            reprojection_loss += self.reprojection_img_loss(params, self.m[i], self.M, rect_Rt[i], rect_img)
            # print(reprojection_loss)

            if(self.visualize):
                cv2.imshow("Image", img)
                cv2.imshow("Rectified Image", rect_img)
                cv2.waitKey()

            if(not os.path.exists(self.result_dir)):
                os.makedirs(self.result_dir, exist_ok=True)
            cv2.imwrite(os.path.join(self.result_dir, 'rectified_' + str(i) + '.png'), rect_img)

        reprojection_error = reprojection_loss/(len(self.m)*len(self.m[0]))
        return reprojection_error

    def calibrate(self):
        print("\nEstimating initial calibration matrix...")
        self.K = self.estimate_intrinsics()
        print("\nInitial Calbiration Matrix, K:\n", self.K)

        print("\nEstimating estimating extrinsics...")
        self.Rt = self.estimate_extrinsics_set(self.K)

        print("\nOptimizing parameters...")
        rect_K, kc = self.optimize_params()
        print("\nOptimized calibration matrix, K:\n", rect_K)
        print("\nDistortion coefficients, kc:\n", kc[:3])

        print("\nComputing reprojection error...")
        rpr_err = self.reprojection_error(rect_K, kc)
        print("\rReprojection Error: ", rpr_err)

        print("Rectified images stored at: ", self.result_dir)