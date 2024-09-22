#!/usr/bin/env python
# coding: utf-8

# In[1]:


# An assignment to implelemtnt YOLO4 for wind turbine surface damage detection.
# Based on paper "Wind Turbine Surface Damage Detection by Deep Learning Aided Drone Inspection Analysis", MDPI, 2019
# Author: Mahmood Karimi
# Student ID: 400115146
# Date: 2022/06/23
# Training Part


# In[2]:


# mount the data set from google drive
from google.colab import drive
drive.mount('/content/gdrive')


# In[3]:


get_ipython().run_line_magic('cd', '/content/gdrive/MyDrive/YOLOv4/darknet/')


# In[ ]:


# create train.txt and test.txt

import glob
import os
import numpy as np
import sys

current_dir = "data/damagedata"
split_pct = 10;
file_train = open("data/train.txt", "w")  
file_val = open("data/test.txt", "w")  
counter = 1  
index_test = round(100 / split_pct)  

i = 0
for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.txt")):  
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        # print('index: ', i, 'title: ', title, ' ext: ', ext)
        i = i + 1
        if counter == index_test:
                counter = 1
                file_val.write(current_dir + "/" + title + '.png' + "\n")
        else:
                file_train.write(current_dir + "/" + title + '.png' + "\n")
                counter = counter + 1
file_train.close()
file_val.close()
print(i, 'samples processed.')


# In[ ]:


# get initial weights
get_ipython().system('wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137')


# In[ ]:


# remove -dont_show to see the the progress chart of mAP-loss against iterations. This should be used when running on colab
# Stop the training when the average loss is less than 0.05 if possible or at least constantly below 0.3, else train the model until the average loss does not show any significant change for a while.
# To restart training from where it stopped, run the following command:
#   darknet.exe detector train data/obj.data cfg/yolov4-custom.cfg ../training/yolov4-custom_last.weights -dont_show -map
get_ipython().system('chmod +x ./darknet')
get_ipython().system('./darknet detector train data/obj.data cfg/yolov4-train.cfg -map -dont_show')


# In[4]:


get_ipython().system('chmod +x ./darknet')
get_ipython().system('./darknet detector train data/obj.data cfg/yolov4-train.cfg ./backup/yolov4-train_last.weights -map -dont_show')


# In[ ]:





# In[ ]:




