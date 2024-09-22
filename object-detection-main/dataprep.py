#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# An assignment to implelemtnt YOLO4 for wind turbine surface damage detection.
# Based on paper "Wind Turbine Surface Damage Detection by Deep Learning Aided Drone Inspection Analysis", MDPI, 2019
# Author: Mahmood Karimi
# Student ID: 400115146
# Date: 2022/06/23
# Data Preparation Part


# In[ ]:


# mount the data set from google drive
from google.colab import drive
drive.mount('/content/gdrive')


# In[1]:


get_ipython().run_line_magic('cd', '/content/gdrive/MyDrive/YOLOv4/')


# In[3]:


# Extract the data file which its link is located in my google drive

import zipfile

with zipfile.ZipFile('/content/gdrive/MyDrive/YOLOv4/NordTank586x371.zip', 'r') as zip_ref:
  zip_ref.extractall('/content/gdrive/MyDrive/YOLOv4/NordTank-data')


# In[ ]:


# download the tool to drow bounding box for existing annotations of test images
get_ipython().system('git clone https://github.com/waittim/draw-YOLO-box')


# In[ ]:


# run the tool to draw bounding boxes for test images
# customize and copy data to the tool before execution
get_ipython().run_line_magic('cd', '/content/gdrive/MyDrive/YOLOv4/draw-YOLO-box/')
get_ipython().system('python draw_box.py')


# In[ ]:





# In[ ]:




