import keras
import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras.models import model_from_yaml
from keras.utils import to_categorical, np_utils
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

np.random.seed(55) # for reproducibility
num_classes = 12
num_epochs = 150

img_rows, img_cols = 100, 100

# Need to reshape data based on backend preferred image format (TF vs Theano)
if K.image_data_format() == 'channels_first':
	x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols).astype('float32')
	x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols).astype('float32')
	input_shape = (1, img_rows, img_cols)
else:
	x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1).astype('float32')
	x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1).astype('float32')
	input_shape = (img_rows, img_cols, 1)

