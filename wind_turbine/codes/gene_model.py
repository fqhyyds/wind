# import the necessary packages
from keras_applications.resnet import ResNet50
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras import regularizers
from sklearn.model_selection import train_test_split
import keras.backend, keras.layers, keras.models, keras.utils
import numpy as np
import cv2
import matplotlib.pyplot as plt


def extract_foreground(image, rect):
	mask = np.zeros(img.shape[:2], np.uint8)

	bgdModel = np.zeros((1, 65), np.float64)
	fgdModel = np.zeros((1, 65), np.float64)

	rect = (77, 0, 98, 224)

	print("[INFO] Extracting Foreground...")
	cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)

	mask2 = np.where((mask == 0) | (mask == 2), 0, 1).astype('uint8')

	return img = img * mask2[:, :, np.newaxis]


def preprocess_images():
	# load the images into memory
	train_images = np.load('train_images.npy')
	coord = np.load('coord.npy')

	train_images_processed = []
	count = 1
	# resize the images to fit the input shape
	# of resnet (224, 224)
	for (i, image) in enumerate(train_images):
		resized_image = cv2.resize(image, (224, 224), cv2.INTER_AREA)
		foreground_extracted_image = extract_foreground(image, coord[i])
		train_images_processes.append(resized_image)
		print("Resized image {}/{}".format(count, train_images.shape[0]))
		count = count + 1

	np.save('train_images_preprocessed.npy', np.array(train_images_preprocessed))


def build_model():
	base_model = ResNet50(include_top=False, backend=keras.backend, layers=keras.layers, models=keras.models,
						  utils=keras.utils, input_shape=(224, 224, 3))

	model = Sequential()
	model.add(base_model)
	model.add(Flatten())
	model.add(Dense(units=32, activation='relu', kernel_initializer='he_normal'))
	model.add(Dense(units=1, activation='sigmoid', kernel_initializer='he_normal'))

	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	print(model.summary())

	return model


def load_data():
	# load the data into memory and split the dataset into train and test splits
	train_images = np.load('train_images_preprocessed.npy')
	train_labels = np.load('train_labels.npy')

	x_train, x_test, y_train, y_test = train_test_split(train_images, train_labels, test_size=0.15, random_state=100)

	return x_train, x_test, y_train, y_test


import_data_to_runtime()
preprocess_images()
model = build_model()
x_train, x_test, y_train, y_test = load_data()
History = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test))


def wrong_predictions():
	# function returns indices of images that the model predicted wrong
	y_test2 = np.reshape(y_test, (y_test.shape[0], 1))
	y_pred = model.predict(x_test)
	result = np.array(y_pred >= 0.5)
	indices = np.where(result != y_test2)

	return indices
