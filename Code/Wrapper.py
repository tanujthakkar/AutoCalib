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
from calibrate_camera import CalibrateCamera


def main():
	Parser = argparse.ArgumentParser()
	Parser.add_argument('--DataDir', type=str, default="../Data/Calibration_Imgs", help='Path to calibration images folder')
	Parser.add_argument('--ResultDir', type=str, default="../Results/", help='Path to results folder')
	Parser.add_argument('--Visualize', action='store_true', help='Toggle visualization')

	Args = Parser.parse_args()
	data_dir = Args.DataDir
	result_dir = Args.ResultDir
	visualize = Args.Visualize

	CC = CalibrateCamera(data_dir, result_dir, visualize)
	CC.calibrate()

if __name__ == '__main__':
	main()