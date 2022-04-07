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

sys.dont_write_bytecode = True

def normalize(img: np.array, _min, _max) -> np.array:
    return np.uint8(cv2.normalize(img, None, _min, _max, cv2.NORM_MINMAX))

def convert_three_channel(img: np.array) -> np.array:
    return np.dstack((img, img, img))

def read_image_set(data_dir: str) -> list:
    return [os.path.join(data_dir,  f) for f in sorted(os.listdir(data_dir))]

def create_image_set(data_dir: str) -> np.array:
	img_set_paths = read_image_set(data_dir)

	img_set = list()
	for img_path in img_set_paths:
		img_set.append(cv2.imread(img_path))

	img_set = np.array(img_set)

	return img_set