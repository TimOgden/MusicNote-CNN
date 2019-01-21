#import keras
import imageio
import glob
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
#start_pixel, end_pixel = 0, 0
start_pixel = 74
end_pixel = 569
horz_pixel = 96

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


def normalize(array):
	newarray = []
	i = 0
	for rgb in array:
		r = rgb[0]
		g = rgb[1]
		b = rgb[2]
		#a = rgb[3]
		val = g / 255
		newarray.append(val)
		i+=1
	return newarray

values = getImportantPixels(numImgs=2)



def plot_values(values):
	for value in values:
		counter = 0
		for val in value:
			#print(counter,':', val)
			plt.plot(counter, val)
			if val >= mean(value):
				plt.scatter(counter, val, color='g')
			else:
				plt.scatter(counter, val, color='b')
			plt.xlabel('Pixel')
			plt.ylabel('Normalized Green Channel')
			counter+=1
		plt.plot(range(len(value)), value, color='k')
		plt.axhline(mean(value))
		plt.show()