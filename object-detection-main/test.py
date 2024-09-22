#!/usr/bin/env python
# coding: utf-8

# In[1]:


# An assignment to implelemtnt YOLO4 for wind turbine surface damage detection.
# Based on paper "Wind Turbine Surface Damage Detection by Deep Learning Aided Drone Inspection Analysis", MDPI, 2019
# Author: Mahmood Karimi
# Student ID: 400115146
# Date: 2022/06/23
# Testing Part


# In[2]:


# mount the data set from google drive
from google.colab import drive
drive.mount('/content/gdrive')


# In[4]:


get_ipython().run_line_magic('cd', '/content/gdrive/MyDrive/YOLOv4/darknet/')
get_ipython().system('chmod +x ./darknet')


# In[ ]:


# Run detector to detect objects in an image
get_ipython().system('./darknet detector test data/obj.data cfg/yolov4-test.cfg backup/yolov4-train_best.weights data/testdata/DJI_0996_03_04.png -thresh 0.3 -dont_show -map')


# In[5]:


get_ipython().system('./darknet detector test data/obj.data cfg/yolov4-test.cfg backup/yolov4-train_best.weights data/testdata/DJI_0996_03_05.png -thresh 0.3 -dont_show -map')


# In[6]:


get_ipython().system('./darknet detector test data/obj.data cfg/yolov4-test.cfg backup/yolov4-train_best.weights data/testdata/DJI_0996_03_06.png -thresh 0.3 -dont_show -map')


# In[7]:


get_ipython().system('./darknet detector test data/obj.data cfg/yolov4-test.cfg backup/yolov4-train_best.weights data/testdata/DJI_0996_05_05.png -thresh 0.3 -dont_show -map')


# In[8]:


get_ipython().system('./darknet detector test data/obj.data cfg/yolov4-test.cfg backup/yolov4-train_best.weights data/testdata/DJI_0996_06_05.png -thresh 0.3 -dont_show -map')


# In[9]:


get_ipython().system('./darknet detector test data/obj.data cfg/yolov4-test.cfg backup/yolov4-train_best.weights data/testdata/DJI_0996_07_05.png -thresh 0.3 -dont_show -map')


# In[10]:


get_ipython().system('./darknet detector test data/obj.data cfg/yolov4-test.cfg backup/yolov4-train_best.weights data/testdata/DJI_0996_08_05.png -thresh 0.3 -dont_show -map')


# In[11]:


get_ipython().system('./darknet detector test data/obj.data cfg/yolov4-test.cfg backup/yolov4-train_best.weights data/testdata/DJI_0997_03_04.png -thresh 0.3 -dont_show -map')


# In[12]:


get_ipython().system('./darknet detector test data/obj.data cfg/yolov4-test.cfg backup/yolov4-train_best.weights data/testdata/DJI_0997_03_05.png -thresh 0.3 -dont_show -map')


# In[13]:


get_ipython().system('./darknet detector test data/obj.data cfg/yolov4-test.cfg backup/yolov4-train_best.weights data/testdata/DJI_0997_08_05.png -thresh 0.3 -dont_show -map')

