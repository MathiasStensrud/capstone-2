# import imageio as io
# import numpy as np
# from scipy.misc import imresize
# import cv2
#
# a=np.array(cv2.LoadImage('img/a1.jpg'))
# b=a[:160, 160:480]
# cv2.ShowImage(b)
# cv2.WaitKey(0)
# cv2.DestroyWindow(name)
# cv2.imwrite('img/')
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

    nb_filters = 32
    kernel_size = (3, 3)
    pool_size = (2, 2)

    model = Sequential()
    # 2 convolutional layers followed by a pooling layer followed by dropout
    model.add(Convolution2D(nb_filters, kernel_size,
                            padding='valid',
                            input_shape=input_size))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, kernel_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))
    # transition to an mlp
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_categories))
    model.add(Activation('softmax'))
    return model

if __name__=='__main__':
    model=create_model([200,200,3],26)
    train=ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory('data/asl_alphabet_train',target_size=[200,200],batch_size=26)
    validate=ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory('data/asl_alphabet_test',[200,200],batch_size=26)
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    callbacks = ModelCheckpoint('./logs',save_best_only=True)
    model.fit_generator(train, epochs=3, steps_per_epoch=2500,validation_data=validate, validation_steps=1, callbacks=[callbacks])

    # best_model = load_model('./logs')
    # holdout_folder  = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory('../data/holdout_small',[100,100],batch_size=16)
    # metrics = best_model.evaluate_generator(holdout_folder, steps=11)
    # print(metrics)
