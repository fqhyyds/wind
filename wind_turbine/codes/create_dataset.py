# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import cv2
import os

data_desc = pd.read_csv('../dataset_desc/desc_csv.csv')
image_name_list = data_desc['Image']
damage = data_desc['Damage']
images = []
save_path = '../dataset/image'
count = 0
for image_name in image_name_list:
	# image = cv2.imread('../dataset/Nordtank_2017/' + image_name)
	image = cv2.imread('../dataset/Nordtank_2018/' + image_name)
	image = cv2.resize(image, (224, 224), cv2.INTER_AREA)
	image = image / 255.0
	images.append(image)
	count = count + 1
	print('Imported Image %d' % count)

images = np.array(images).astype(np.float16)
print('Saving to disk...')
save_full_images_path = os.path.join(save_path, 'train_images.npy')
save_full_labels_path = os.path.join(save_path, 'train_labels.npy')
np.save('train_images.npy', images)
np.save('train_labels.npy', damage)
