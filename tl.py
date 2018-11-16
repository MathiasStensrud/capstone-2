from keras.layers import Activation, Convolution2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D
from keras.models import Sequential, Model
from keras.applications.xception import preprocess_input
from keras.applications import Xception, InceptionV3
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import load_model

def create_transfer_model(input_size, n_categories, weights = 'imagenet'):

        base_model = Xception(weights=weights,
                          include_top=False,
                          input_shape=input_size)

        model = base_model.output
        model = GlobalAveragePooling2D()(model)
        predictions = Dense(n_categories, activation='softmax')(model)
        model = Model(inputs=base_model.input, outputs=predictions)

        return model

def change_trainable_layers(model, trainable_index):

    for layer in model.layers[:trainable_index]:
        layer.trainable = False
    for layer in model.layers[trainable_index:]:
        layer.trainable = True



if __name__=='__main__':
    model=create_transfer_model([75,75,3],5)
    train=ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory('data/holdout',target_size=[75,75],batch_size=27)
    validate=ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory('data/train2/asl_alphabet_test',[75,75],batch_size=27)
    tensor=TensorBoard(log_dir='./tens_logs', histogram_freq=0, batch_size=27, write_graph=True, write_grads=False, write_images=False)
    callbacks = ModelCheckpoint('./last_tl_log',save_best_only=True)
    # holdout_folder  = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory('../data/holdout_small',[100,100],batch_size=16)
    # metrics = best_model.evaluate_generator(holdout_folder, steps=11)
    # print(metrics)
    trans_model = create_transfer_model((75,75,3),27)
    _ = change_trainable_layers(trans_model, 132)
    trans_model.compile(optimizer=RMSprop(lr=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
    trans_model.fit_generator(train, epochs=10, steps_per_epoch=30, validation_data=validate,validation_steps=5, callbacks=[callbacks, tensor])
    _ = change_trainable_layers(trans_model, 126)
    trans_model.compile(optimizer=RMSprop(lr=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
    trans_model.fit_generator(train, epochs=50, steps_per_epoch=30, validation_data=validate,validation_steps=5, callbacks=[callbacks, tensor])
