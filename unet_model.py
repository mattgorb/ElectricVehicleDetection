import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import concatenate,BatchNormalization,Activation, Dense, Conv1D,MaxPooling1D,  Input,  UpSampling1D, Dropout, LSTM
from keras.losses import binary_crossentropy
import keras.backend as K
from keras.optimizers import RMSprop, SGD
from keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator
import math


def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score


def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss
def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss

def get_unet(num_classes=1):
    inputs = Input((2880,1))
    # 1024
			#8
    '''down0b = Conv1D(1, 3, padding='same')(inputs)
    down0b = BatchNormalization()(down0b)
    down0b = Activation('relu')(down0b)
    down0b = Conv1D(1, 3, padding='same')(down0b)
    down0b = BatchNormalization()(down0b)
    down0b = Activation('relu')(down0b)
    down0b_pool = MaxPooling1D(1, strides=1)(down0b)
    # 512

    down0a = Conv1D(2, 3, padding='same')(down0b_pool)
    down0a = BatchNormalization()(down0a)
    down0a = Activation('relu')(down0a)
    down0a = Conv1D(2, 3, padding='same')(down0a)
    down0a = BatchNormalization()(down0a)
    down0a = Activation('relu')(down0a)
    down0a_pool = MaxPooling1D(1, strides=2)(down0a)'''
    # 256

    down0 = Conv1D(4, 3, padding='same')(inputs)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0 = Conv1D(4, 3, padding='same')(down0)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0_pool = MaxPooling1D(1, strides=2)(down0)
    # 128

    down1 = Conv1D(8, 3, padding='same')(down0_pool)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1 = Conv1D(8, 3, padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1_pool = MaxPooling1D(1, strides=2)(down1)
    # 64

    down2 = Conv1D(16, 3, padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2 = Conv1D(16, 3, padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2_pool = MaxPooling1D(1, strides=2)(down2)
    # 32

    down3 = Conv1D(32, 3, padding='same')(down2_pool)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3 = Conv1D(32, 3, padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3_pool = MaxPooling1D(1, strides=2)(down3)
    # 16

    down4 = Conv1D(64, 3, padding='same')(down3_pool)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4 = Conv1D(64, 3, padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4_pool = MaxPooling1D(1, strides=2)(down4)
    # 8

    center = Conv1D(128, 3, padding='same')(down4_pool)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv1D(128, 3, padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    # center

    up4 = UpSampling1D(2)(center)
    up4 = concatenate([down4, up4], axis=2)
    up4 = Conv1D(64, 3, padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv1D(64, 3, padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv1D(64, 3, padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    # 16

    up3 = UpSampling1D(2)(up4)
    up3 = concatenate([down3, up3], axis=2)
    up3 = Conv1D(32, 3, padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv1D(32, 3, padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv1D(32, 3, padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    # 32

    up2 = UpSampling1D(2)(up3)
    up2 =concatenate([down2, up2], axis=2)
    up2 = Conv1D(16, 3, padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv1D(16, 3, padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv1D(16, 3, padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    # 64

    up1 = UpSampling1D(2)(up2)
    up1 = concatenate([down1, up1], axis=2)
    up1 = Conv1D(8, 3, padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv1D(8, 3, padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv1D(8, 3, padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    # 128

    up0 = UpSampling1D(2)(up1)
    up0 = concatenate([down0, up0], axis=2)
    up0 = Conv1D(4, 3, padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Conv1D(4, 3, padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Conv1D(4, 3, padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    # 256

    '''up0a = UpSampling1D(2)(up0)
    up0a = concatenate([down0a, up0a], axis=2)
    up0a = Conv1D(2, 3, padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    up0a = Conv1D(2, 3, padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    up0a = Conv1D(2, 3, padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    # 512

    up0b = UpSampling1D(1)(up0a)
    up0b = concatenate([down0b, up0b], axis=2)
    up0b = Conv1D(1, 3, padding='same')(up0b)
    up0b = BatchNormalization()(up0b)
    up0b = Activation('relu')(up0b)
    up0b = Conv1D(1, 3, padding='same')(up0b)
    up0b = BatchNormalization()(up0b)
    up0b = Activation('relu')(up0b)
    up0b = Conv1D(1, 3, padding='same')(up0b)
    up0b = BatchNormalization()(up0b)
    up0b = Activation('relu')(up0b)'''
    # 1024

    classify = Conv1D(num_classes, 1, activation='sigmoid')(up0)

    model = Model(inputs=inputs, outputs=classify)
    
    model.compile(optimizer=RMSprop(lr=0.01), loss=bce_dice_loss, metrics=[dice_coeff])
    return model
