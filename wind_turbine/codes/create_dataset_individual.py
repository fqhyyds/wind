import os
import numpy as np
import pandas as pd
import cv2

data_desc = pd.read_csv('../dataset_desc/desc_csv.csv')
image_name_list = data_desc['Image']
damage = data_desc['Damage']
save_path = '../dataset/image'


def get_indices():
	images_18 = os.listdir('../dataset/Nordtank_2017')

	indices = []
	for image_name in images_18:
		index = np.where(image_name_list == image_name)
		indices.append(index)
	indices = np.squeeze(np.array(indices))

	return indices


def load_images_and_labels():

	indices = get_indices()
	image_name_list_18 = image_name_list[indices]

	images = []
	count = 1
	for image_name in image_name_list_18:
		image = cv2.imread('../dataset/Nordtank_2017/' + image_name)
		image = cv2.resize(image, (224, 224), cv2.INTER_AREA)
		image = image / 255.0
		images.append(image)
		print('Imported Image %d' % count)
		count = count + 1

	images = np.array(images).astype(np.float16)
	labels = damage[indices]

	return images, labels


images, labels = load_images_and_labels()

print('Saving to disk...')
save_full_images_path = os.path.join(save_path, 'train_images_17.npy')
save_full_labels_path = os.path.join(save_path, 'train_labels_17.npy')
np.save(save_full_images_path, images)
np.save(save_full_labels_path, labels)
