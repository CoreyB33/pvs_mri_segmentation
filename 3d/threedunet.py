from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, AveragePooling3D, concatenate,\
    GlobalAveragePooling3D, add, UpSampling3D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Activation, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import tensorflow as tf
import losses
from losses import *
import multi_gpu
from multi_gpu import ModelMGPU
import json


def threedunet(model_path,
         num_channels,
         loss="binary_crossentropy",
         ds=2,
         lr=1e-4,
         num_gpus=1,
         verbose=0,):
    inputs = Input((None, None, None, num_channels))

    conv1 = Conv3D(32//ds, 3, activation='relu', padding='same', )(inputs)
    conv1 = Conv3D(64//ds, 3, activation='relu', padding='same', )(conv1) 
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(64//ds, 3, activation='relu', padding='same',)(pool1)
    conv2 = Conv3D(128//ds, 3, activation='relu', padding='same', )(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(128//ds, 3, activation='relu', padding='same', )(pool2)
    conv3 = Conv3D(256//ds, 3, activation='relu', padding='same', )(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = Conv3D(256//ds, 3, activation='relu', padding='same', )(pool3)
    conv4 = Conv3D(512//ds, 3, activation='relu', padding='same', )(conv4)

    up5 = Conv3D(512//ds, 2, strides=1, activation='relu', padding='same')(UpSampling3D(size=(2, 2, 2))(conv4))
    merge5 = concatenate([conv3, up5], axis=4)
    conv5 = Conv3D(256//ds, 3, strides=1, activation='relu', padding='same')(merge5)
    conv5 = Conv3D(256//ds, 3, strides=1, activation='relu', padding='same')(conv5)

    up6 = Conv3D(256//ds, 2, strides=1, activation='relu', padding='same')(UpSampling3D(size=(2, 2, 2))(conv5))
    merge6 = concatenate([conv2, up6], axis=4)
    conv6 = Conv3D(128//ds, 3, strides=1, activation='relu', padding='same')(merge6)
    conv6 = Conv3D(128//ds, 3, strides=1, activation='relu', padding='same')(conv6)

    up7 = Conv3D(128//ds, 2, strides=1, activation='relu', padding='same')(UpSampling3D(size=(2, 2, 2))(conv6))
    merge7 = concatenate([conv1, up7], axis=4)
    conv7 = Conv3D(64//ds, 3, strides=1, activation='relu', padding='same')(merge7)
    conv7 = Conv3D(64//ds, 3, strides=1, activation='relu', padding='same')(conv7)

    conv8 = Conv3D(2, 3, strides=1, activation='relu', padding='same', )(conv7)
    conv9 = Conv3D(1, 1, activation='sigmoid')(conv8)

    model = tf.keras.Model(inputs=inputs, outputs=conv9)

    # dice as a human-readble metric
    model.compile(optimizer=Adam(lr=lr),
                  metrics=[dice_coef],
                  loss=loss)

    # save json before checking if multi-gpu
    json_string = model.to_json()
    with open(model_path, 'w') as f:
        json.dump(json_string, f)

    if verbose:
        print(model.summary())

    # recompile if multi-gpu model
    if num_gpus > 1:
        model = ModelMGPU(model, num_gpus)
        model.compile(optimizer=Adam(lr=lr),
                      metrics=[dice_coef],
                      loss=loss)

    return model
