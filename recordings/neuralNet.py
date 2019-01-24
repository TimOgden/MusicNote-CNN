import keras
import imageio
import glob, os
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
#start_pixel, end_pixel = 0, 0
start_pixel = 5
end_pixel = 374
horz_pixel = 96

training_data = {}
testing_data = {}
labels = {}

def getImportantPixels(numImgs=None):
	currentImage = 0
	listOfPixelArrays = []
	for path in glob.glob("*.png"):
		if numImgs is not None:
			if currentImage>=numImgs:
				break

		img = imageio.imread(path)
		np_img = np.array(img, dtype='int32')
		array = []
		for c in range(start_pixel, end_pixel):
			r = np_img[horz_pixel][c][0]
			g = np_img[horz_pixel][c][1]
			b = np_img[horz_pixel][c][2]
			a = np_img[horz_pixel][c][3]
			array.append([r,g,b,a])
		np_array = np.array(array)
		np_array = normalize(np_array)
		listOfPixelArrays.append(np_array)
		currentImage+=1
	return listOfPixelArrays

def getImportantPixels(test_split=.8):
	currentImage = 0
	numImgs = len(os.listdir()) * test_split
	goesToTraining = True
	listOfPixelArrays = []
	for path in glob.glob("*.png"):
		if numImgs is not None:
			if currentImage>=numImgs:
				goesToTraining = False

		img = imageio.imread(path)
		np_img = np.array(img, dtype='int32')
		array = []
		for c in range(start_pixel, end_pixel):
			green_row_vals = []
			for h in range(65, 500):
				r = np_img[h][c][0]
				g = np_img[h][c][1]
				b = np_img[h][c][2]
				a = np_img[h][c][3]
				green_row_vals.append(g)
			array.append(mean(green_row_vals))
		np_array = np.array(array)
		np_array = normalize(np_array)

		if goesToTraining:
			training_data[removeFileName(path)] = np_array
		else:
			testing_data[removeFileName(path)] = np_array

		listOfPixelArrays.append(np_array)
		currentImage+=1
	return listOfPixelArrays

def getImportantPixels(filename):
	img = imageio.imread(filename)
	np_img = np.array(img, dtype='int32')
	array = []
	for c in range(start_pixel, end_pixel):
		r = np_img[horz_pixel][c][0]
		g = np_img[horz_pixel][c][1]
		b = np_img[horz_pixel][c][2]
		a = np_img[horz_pixel][c][3]
		array.append([r,g,b,a])
	np_array = np.array(array)
	np_array = normalize(np_array)
	return np_array

def removeFileName(path):
	index = path.index('.')
	return path[:index]


def normalize(array):
	newarray = []
	i = 0
	for rgb in array:
		#r = rgb[0]
		g = rgb[1]
		#b = rgb[2]
		#a = rgb[3]
		val = g / 255
		newarray.append(val)
		i+=1
	return newarray




def plot_values(values):
	counter = 0
	for val in values:
		#print(counter,':', val)
		plt.plot(counter, val)
		if val >= mean(values):
			plt.scatter(counter, val, color='g')
		else:
			plt.scatter(counter, val, color='b')
		plt.xlabel('Pixel')
		plt.ylabel('Normalized Green Channel')
		counter+=1
	plt.plot(range(len(values)), values, color='k')
	plt.axhline(mean(values))
	plt.show()

'''
values1 = getImportantPixels('C#-9.png')
values2 = getImportantPixels('G-22.png') 
plot_values(values1)
plot_values(values2)
'''

def build_model():
	model = keras.models.Sequential([
		keras.layers.Dense(495, activation='relu', input_shape=(495,)), #Input layer
		keras.layers.Dropout(.2),
		keras.layers.Dense(256, activation='relu'),
		keras.layers.Dropout(.2),
		keras.layers.Dense(128, activation='relu'),
		keras.layers.Dropout(.1),
		keras.layers.Dense(128, activation='relu'),
		keras.layers.Dropout(.1),
		keras.layers.Dense(56, activation='softmax') # Output layer
	])
	model.compile(optimizer=keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy','mse'])
	return model

model = build_model()

model.fit()
