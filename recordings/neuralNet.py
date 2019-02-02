import keras
import imageio
import glob, os, sys
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
from tqdm import tqdm

#start_pixel, end_pixel = 0, 0
start_pixel = 5
end_pixel = 374
horz_pixel = 96

loadInData = True


chords = {}

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

def getData(test_split=.8):
	training_data_y = []
	training_data_x = []
	testing_data_y = []
	testing_data_x = []
	currentImage = 0
	numImgs = len(os.listdir()) * test_split
	goesToTraining = True
	listOfPixelArrays = []
	for path in glob.glob(os.path.join("spectrograms","*.png")):
		if numImgs is not None:
			if currentImage>=numImgs:
				goesToTraining = False

		img = imageio.imread(path)
		np_img = np.array(img, dtype='int32')
		array = []
		for c in range(start_pixel, end_pixel):
			green_row_vals = []
			for h in range(65, 500):
				try:
					#r = np_img[h][c][0]
					g = np_img[h][c][1]
					#b = np_img[h][c][2]
					#a = np_img[h][c][3]
				except:
					break
				green_row_vals.append(g)
			array.append(mean(green_row_vals))
		array = normalize(array)
		np_array = np.array(array)

		#print(path, np_array.shape)
		if goesToTraining:
			training_data_y.append(removeFileName(path))
			training_data_x.append(np_array)
		else:
			testing_data_y.append(removeFileName(path))
			testing_data_x.append(np_array)

		currentImage+=1
		onehotConversion(training_data_y)
		onehotConversion(testing_data_y)
	return (training_data_x, training_data_y), (testing_data_x, testing_data_y)

def getImportantPixels(path):
	img = imageio.imread(path)
	np_img = np.array(img, dtype='int32')
	array = []
	for c in range(start_pixel, end_pixel):
		green_row_vals = []
		for h in range(65, 500):
			try:
				#r = np_img[h][c][0]
				g = np_img[h][c][1]
				#b = np_img[h][c][2]
				#a = np_img[h][c][3]
			except:
				break
			green_row_vals.append(g)
		array.append(mean(green_row_vals))
	array = normalize(array)
	np_array = np.array(array)

	return np_array

def removeFileName(path):
	index = path.index('-')
	return path[:index]


def normalize(array):
	newarray = []
	i = 0
	for g in array:
		#r = rgb[0]
		#b = rgb[2]
		#a = rgb[3]
		val = g / 255
		newarray.append(val)
		i+=1
	return newarray

chord_customizers = {'root_note': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
						'chord_type': ['major', 'minor', 'major 7th', 'minor 7th',
									 'sus2', 'sus4', '8th interval', 'fifth interval']} 

def onehotConversion(train):
	for c, val in enumerate(train):
		i = 0
		if type(val)==str:
			val = val[val.index('\\')+1:]
			for note in chord_customizers['root_note']:
				for chord_type in chord_customizers['chord_type']:
					#print(val[0], val[1:])
					if val[0] == note and val[1:] == chord_type:
						train[c] = np.zeros(56)
						train[c][i] = 1
						#print(val, ':', i)
						chords[i] = val
						#print('Converting to onehot:', train[c])
					i+=1
	return train


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

def plot(arr, title):
	for c, element in enumerate(arr):
		plt.bar(c, element, color='b')
	plt.title(title)
	plt.show()


def build_model():
	model = keras.models.Sequential([
		keras.layers.Dense(369, activation='relu', input_shape=(369,)), #Input layer
		keras.layers.Dropout(.2),
		keras.layers.Dense(256, activation='relu'),
		keras.layers.Dropout(.2),
		keras.layers.Dense(128, activation='relu'),
		keras.layers.Dropout(.1),
		keras.layers.Dense(128, activation='relu'),
		keras.layers.Dropout(.1),
		keras.layers.Dense(56, activation='softmax') # Output layer
	])
	model.compile(optimizer=keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	return model

model = build_model()
print('model built')

train_x = []
train_y = []
test_x = []
test_y = []


#plot(getImportantPixels('C:/Users/Tim/ProgrammingProjects/MusicNote-CNN/recordings/spectrograms/A8th interval-1.png'), 'A8th interval')
#plot(getImportantPixels('C:/Users/Tim/ProgrammingProjects/MusicNote-CNN/recordings/spectrograms/Gsus2-1.png'), 'Gsus2')


if not loadInData:
	(train_x, train_y), (test_x, test_y) = getData(test_split=.8)
	np.save('train_x.npy', train_x)
	np.save('train_y.npy', train_y)
	np.save('test_x.npy', test_x)
	np.save('test_y.npy', test_y)
	print('data gathered for first time')
else:
	train_x = np.load('train_x.npy')
	train_y = np.load('train_y.npy')
	test_x = np.load('test_x.npy')
	test_y = np.load('test_y.npy')
	print('data gathered from save')

train_x = np.array(train_x)
train_y = np.array(train_y)
test_x = np.array(test_x)
test_y = np.array(test_y)
print(train_x.shape)
print(train_y.shape)
print('fitting network now')
model.fit(x=train_x, y=train_y, epochs=5, verbose=2)

print('saving weights')
model.save('neural_net-epoch5.model')

predictions = model.predict(test_x)
for prediction in predictions:
	print(chords[np.argmax(prediction)])
