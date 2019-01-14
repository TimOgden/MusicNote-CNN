import keras
import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras.models import model_from_yaml
from keras.utils import to_categorical, np_utils
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.preprocessing import image

np.random.seed(55) # for reproducibility
num_classes = 12
num_epochs = 150

img_rows, img_cols = 100, 100

training_data = np.array()
for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
		training_data.append(filename)
print(training_data)