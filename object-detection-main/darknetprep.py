#!/usr/bin/env python
# coding: utf-8
import os
import subprocess

# An assignment to implelemtnt YOLO4 for wind turbine surface damage detection.
# Based on paper "Wind Turbine Surface Damage Detection by Deep Learning Aided Drone Inspection Analysis", MDPI, 2019
# Author: Mahmood Karimi
# Student ID: 400115146
# Date: 2022/06/23
# Darknet (YOLOv4 implementation) Preparation Part

# mount the data set from google drive
# from google.colab import drive
# drive.mount('/content/gdrive')
# get_ipython().run_line_magic('cd', '/content/gdrive/MyDrive/YOLOv4/')
# 请将以下路径更改为您本地的darknet文件夹路径


# Get a copy of Darknet repository
darknet_path = '../object-detection-main/darknet/'
# if not os.path.exists(darknet_path):
#     subprocess.run(['git', 'clone', 'https://github.com/AlexeyAB/darknet', darknet_path])

# get_ipython().system('git clone https://github.com/AlexeyAB/darknet')

# Navigate to the darknet directory
# 存储了darknet仓库在本地文件系统中的路径
os.chdir(darknet_path)

# Replace parameters in Makefile to use be used on COLAB
# 将Makefile中所有出现的OPENCV=0替换为OPENCV=1,启用了OpenCV库
subprocess.run(['sed', '-i', 's/OPENCV=0/OPENCV=1/', 'Makefile'])
# 启用了GPU加速
subprocess.run(['sed', '-i', 's/GPU=0/GPU=1/', 'Makefile'])
# CUDNN是NVIDIA提供的用于深度神经网络的GPU加速库
subprocess.run(['sed', '-i', 's/CUDNN=0/CUDNN=1/', 'Makefile'])
# 启用了CUDNN的半精度支持，它可以减少内存使用并可能提高性能，尤其是在使用支持半精度计算的GPU时
subprocess.run(['sed', '-i', 's/CUDNN_HALF=0/CUDNN_HALF=1/', 'Makefile'])
# 共享内存并行编程的API，它可以使得代码在多核处理器上运行得更快
subprocess.run(['sed', '-i', 's/OPENMP=0/OPENMP=1/', 'Makefile'])
# get_ipython().run_line_magic('cd', 'darknet')
# get_ipython().system("sed -i 's/OPENCV=0/OPENCV=1/' Makefile")
# get_ipython().system("sed -i 's/GPU=0/GPU=1/' Makefile")
# get_ipython().system("sed -i 's/CUDNN=0/CUDNN=1/' Makefile")
# get_ipython().system("sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile")
# get_ipython().system("sed -i 's/OPENMP=0/OPENMP=1/' Makefile")


# Make the Darknet which is the core CNN stack used in YOLO
# get_ipython().system('make')
subprocess.run(['make'])

# Make it executable
# get_ipython().system('chmod +x ./darknet')
subprocess.run(['chmod', '+x', '/darknet'])