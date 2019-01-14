from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
	width_shift_range=0.5,
	fill_mode='nearest')