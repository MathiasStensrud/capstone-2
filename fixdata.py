import imageio as io
import numpy as np
from scipy.misc import imresize
import cv2 as cv
import skimage
from keras.layers import Activation, Convolution2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D
from keras.models import Sequential, Model
from keras.applications.xception import preprocess_input
from keras.applications import Xception
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.models import load_model



def create_model(input_size, n_categories):
    """
    Create a simple baseline CNN

    Args:
        input_size (tuple(int, int, int)): 3-dimensional size of input to model
        n_categories (int): number of classification categories

    Returns:
        keras Sequential model: model with new head
        """

    nb_filters = 64
    kernel_size = (5, 5)
    pool_size = (2, 2)

    model = Sequential()
    model.add(Convolution2D(nb_filters, kernel_size,
                            padding='valid',
                            input_shape=input_size))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, kernel_size))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Convolution2D(nb_filters, kernel_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.5))
    model.add(Convolution2D(nb_filters, kernel_size))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, kernel_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_categories))
    model.add(Activation('softmax'))
    return model



if __name__=='__main__':
    model=create_model([72,72,3],27)
    train=ImageDataGenerator(preprocessing_function=preprocess_input, horizontal_flip=True).flow_from_directory('data/holdout',target_size=[72,72],batch_size=27)
    validate=ImageDataGenerator(preprocessing_function=preprocess_input, horizontal_flip=True).flow_from_directory('data/train2/asl_alphabet_test',[72,72],batch_size=27)
    # holdout=ImageDataGenerator(preprocessing_function=preprocess_input, horizontal_flip=True).flow_from_directory('data/holdout',[72,72],batch_size=30)
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    callbacks = ModelCheckpoint('./logs2',save_best_only=True)
    model.fit_generator(train, epochs=10,steps_per_epoch=100, validation_data=validate,validation_steps=10, callbacks=[callbacks])
    # best_model=load_model('./logs')
    # print(best_model.metric_names, '\n',best_model.evaluate_generator(validate, steps=27))
