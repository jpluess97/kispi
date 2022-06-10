import tensorflow as tf
from keras import backend as K
from keras.models import Model, Sequential, load_model
import torch.nn as nn
from keras.layers import Conv2D, Input, MaxPooling2D, concatenate, Dropout, \
    Lambda, Conv3DTranspose, Add, Dense, BatchNormalization, \
    AveragePooling2D, Conv3D, Conv2DTranspose, Concatenate, \
    Flatten, MaxPooling3D, UpSampling2D, LeakyReLU, GlobalAveragePooling2D, Reshape, Permute, multiply, Activation, \
    Multiply, UpSampling3D

from keras.activations import relu, softmax

from keras.preprocessing.image import img_to_array, load_img
from keras import regularizers
from keras.layers import Input, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, Concatenate, Activation, Add, multiply, add, concatenate, LeakyReLU, ZeroPadding2D, UpSampling2D, BatchNormalization
import os

# making the range (-1, 1)
from scipy.spatial.distance import dice
from tensorflow_addons.layers import InstanceNormalization



def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

# 2D-Unet
def unet_kelly(input_shape, model_name, n_classes):
    
    i = Input((input_shape[0], input_shape[1], input_shape[2]))
    s = Lambda(lambda x: preprocess_input(x)) (i)
    
    #dowsampling part 
    conv1 = Conv2D(32, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01))(s)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01))(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1)
    pool1 = Dropout(0.25)(pool1)
    
    conv2 = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01))(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01))(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2,2))(conv2)
    pool2 = Dropout(0.5)(pool2)
    
    conv3 = Conv2D(128, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01))(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01))(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2,2))(conv3)
    pool3 = Dropout(0.5)(pool3)
    
    conv4 = Conv2D(256, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01))(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01))(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2,2))(conv4)
    pool4 = Dropout(0.5)(pool4)
    
    conv5 = Conv2D(512, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01))(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(512, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01))(conv5)
    conv5 = BatchNormalization()(conv5)
    
    # upsampling part

    up6 = UpSampling2D(size=(2, 2))(conv5)
    up6 = Concatenate(axis=3)([up6, conv4])
    up6 = Dropout(0.5)(up6)
    conv6 = Conv2D(256, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01))(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(256, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01))(conv6)
    conv6 = BatchNormalization()(conv6)
    
    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = Concatenate(axis=3)([up7, conv3])
    up7 = Dropout(0.5)(up7)
    conv7 = Conv2D(128, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01))(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(128, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01))(conv7)
    conv7 = BatchNormalization()(conv7)
       
    up8 = UpSampling2D(size=(2, 2))(conv7)
    up8 = Concatenate(axis=3)([up8, conv2])
    up8 = Dropout(0.5)(up8)
    conv8 = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01))(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01))(conv8)
    conv8 = BatchNormalization()(conv8)
    
    up9 = UpSampling2D(size=(2, 2))(conv8)
    up9 = Concatenate(axis=3)([up9, conv1])
    up9 = Dropout(0.5)(up9)
    conv9 = Conv2D(32, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01), name='last_3')(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(32, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01), name='last_2')(conv9)
    conv9 = BatchNormalization()(conv9)

    pred = Conv2D(2, kernel_size=(1,1),  activation='softmax', padding='valid', name='last_1')(conv9)
    final_model = Model(inputs=i, outputs=pred, name = model_name)
    

    return final_model

def Unet(input_shape, model_name, n_classes, pretrained=False,  base=4):
    assert n_classes > 1, "number classes have to be greater than one!"
    
    if pretrained:
        path = os.path.join('models', model_name+'.model')
        if os.path.exists(path):
            model = load_model(path, custom_objects={'dice': dice})
            model.summary()
            return model
        else:
            print('Failed to load existing model at: {}'.format(path))

    b = base
    i = Input((input_shape[0], input_shape[1], input_shape[2]))
    s = Lambda(lambda x: preprocess_input(x)) (i)

    c1 = Conv2D(2**b, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
    c1 = Dropout(0.1) (c1)
    c1 = Conv2D(2**b, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(2**(b+1), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = Dropout(0.1) (c2)
    c2 = Conv2D(2**(b+1), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(2**(b+2), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = Dropout(0.2) (c3)
    c3 = Conv2D(2**(b+2), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(2**(b+3), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = Dropout(0.2) (c4)
    c4 = Conv2D(2**(b+3), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(2**(b+4), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
    c5 = Dropout(0.3) (c5)
    c5 = Conv2D(2**(b+4), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

    u6 = Conv2DTranspose(2**(b+3), (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(2**(b+3), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = Dropout(0.2) (c6)
    c6 = Conv2D(2**(b+3), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

    u7 = Conv2DTranspose(2**(b+2), (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(2**(b+2), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = Dropout(0.2) (c7)
    c7 = Conv2D(2**(b+2), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

    u8 = Conv2DTranspose(2**(b+1), (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(2**(b+1), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = Dropout(0.1) (c8)
    c8 = Conv2D(2**(b+1), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

    u9 = Conv2DTranspose(2**b, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(2**b, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = Dropout(0.1) (c9)
    c9 = Conv2D(2**b, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

    o = Conv2D(n_classes, (1, 1), activation = "softmax") (c9)

    model = Model(inputs=i, outputs=o, name = model_name)
    return model

    
def unet_3d(input_shape, model_name, n_classes, pretrained=False,  base=4):
    assert n_classes > 1, "number classes have to be greater than one!"
    
    if pretrained:
        path = os.path.join('models', model_name+'.model')
        if os.path.exists(path):
            model = load_model(path, custom_objects={'dice': dice})
            model.summary()
            return model
        else:
            print('Failed to load existing model at: {}'.format(path))

    b = base
    i = Input((input_shape[0], input_shape[1], input_shape[2], input_shape[3]))
    #s = Lambda(lambda x: preprocess_input(x)) (i)

    c1 = Conv3D(2**b, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
    c1 = Dropout(0.1) (c1)
    c1 = Conv3D(2**b, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
    p1 = MaxPooling3D((2, 2, 2)) (c1)

    c2 = Conv3D(2**(b+1), (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = Dropout(0.1) (c2)
    c2 = Conv3D(2**(b+1), (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
    p2 = MaxPooling3D((2, 2, 2)) (c2)

    c3 = Conv3D(2**(b+2), (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv3D(2**(b+2), (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
    p3 = MaxPooling3D((2, 2, 2))(c3)

    c4 = Conv3D(2**(b+3), (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = Dropout(0.2) (c4)
    c4 = Conv3D(2**(b+3), (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
    p4 = MaxPooling3D((2, 2, 2)) (c4)

    c5 = Conv3D(2**(b+4), (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
    c5 = Dropout(0.3) (c5)
    c5 = Conv3D(2**(b+4), (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

    u6 = Conv3DTranspose(2**(b+3), (2, 2, 2), strides=(2, 2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv3D(2**(b+3), (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = Dropout(0.2) (c6)
    c6 = Conv3D(2**(b+3), (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

    u7 = Conv3DTranspose(2**(b+2), (2, 2, 2), strides=(2, 2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv3D(2**(b+2), (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = Dropout(0.2) (c7)
    c7 = Conv3D(2**(b+2), (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

    u8 = Conv3DTranspose(2**(b+1), (2, 2, 2), strides=(2, 2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv3D(2**(b+1), (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = Dropout(0.1) (c8)
    c8 = Conv3D(2**(b+1), (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

    u9 = Conv3DTranspose(2**b, (2, 2, 2), strides=(2, 2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv3D(2**b, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = Dropout(0.1) (c9)
    c9 = Conv3D(2**b, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

    o = Conv3D(n_classes, (1, 1, 1), activation = "softmax") (c9)

    model = Model(inputs=i, outputs=o, name = model_name)
    return model

# UNET MODEL WITH CHANGES
def unet_jaye(input_shape, model_name, n_classes, pretrained = False):
    i = Input((input_shape[0], input_shape[1], input_shape[2]))
    i /= 255.
    i -= 0.5
    i *= 2.
    #s = Lambda(lambda x: preprocess_input(x))(i)

    if pretrained:
        path = os.path.join('models', model_name+'.model')
        if os.path.exists(path):
            model = load_model(path, custom_objects={'dice': dice})
            model.summary()
            return model
        else:
            print('Failed to load existing model at: {}'.format(path))

    conv1 = Conv2D(32, kernel_size=(3, 3), activation=LeakyReLU(), padding='same', kernel_regularizer=regularizers.l2(0.01))(
        s)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, kernel_size=(3, 3), activation=LeakyReLU(), padding='same', kernel_regularizer=regularizers.l2(0.01))(
        conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(0.1)(pool1)

    conv2 = Conv2D(64, kernel_size=(3, 3), activation=LeakyReLU(), padding='same', kernel_regularizer=regularizers.l2(0.01))(
        pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, kernel_size=(3, 3), activation=LeakyReLU(), padding='same', kernel_regularizer=regularizers.l2(0.01))(
        conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(0.1)(pool2)

    conv3 = Conv2D(128, kernel_size=(3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, kernel_size=(3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(0.1)(pool3)

    conv4 = Conv2D(256, kernel_size=(3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, kernel_size=(3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = Dropout(0.1)(pool4)

    conv5 = Conv2D(512, kernel_size=(3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(512, kernel_size=(3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(conv5)
    conv5 = BatchNormalization()(conv5)


    up6 = UpSampling2D(size=(2, 2))(conv5)
    up6 = Concatenate(axis=3)([up6, conv4])
    up6 = Dropout(0.1)(up6)
    conv6 = Conv2D(256, kernel_size=(3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(256, kernel_size=(3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = Concatenate(axis=3)([up7, conv3])
    up7 = Dropout(0.1)(up7)
    conv7 = Conv2D(128, kernel_size=(3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(128, kernel_size=(3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = UpSampling2D(size=(2, 2))(conv7)
    up8 = Concatenate(axis=3)([up8, conv2])
    up8 = Dropout(0.1)(up8)
    conv8 = Conv2D(64, kernel_size=(3, 3), activation=LeakyReLU(), padding='same', kernel_regularizer=regularizers.l2(0.01))(
        up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(64, kernel_size=(3, 3), activation=LeakyReLU(), padding='same', kernel_regularizer=regularizers.l2(0.01))(
        conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = UpSampling2D(size=(2, 2))(conv8)
    up9 = Concatenate(axis=3)([up9, conv1])
    up9 = Dropout(0.1)(up9)
    conv9 = Conv2D(32, kernel_size=(3, 3), activation=LeakyReLU(), padding='same', kernel_regularizer=regularizers.l2(0.01),
                   name='last_3')(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(32, kernel_size=(3, 3), activation=LeakyReLU(), padding='same', kernel_regularizer=regularizers.l2(0.01),
                   name='last_2')(conv9)
    conv9 = BatchNormalization()(conv9)

    pred = Conv2D(n_classes, kernel_size=(1, 1), activation='softmax', padding='valid', name='last_1')(conv9)
    final_model = Model(inputs=i, outputs=pred, name=model_name)

    return final_model

    return x

def unet_jaye_3d_ch(input_shape, model_name, n_classes, pretrained = False):
    i = Input((input_shape[0], input_shape[1], input_shape[2], input_shape[3]))
    i /= 255.
    i -= 0.5
    i *= 2.
    # s = Lambda(lambda x: preprocess_input(x))(i)

    if pretrained:
        path = os.path.join('models', model_name+'.model')
        if os.path.exists(path):
            model = load_model(path, custom_objects={'dice': dice})
            model.summary()
            return model
        else:
            print('Failed to load existing model at: {}'.format(path))

    conv1 = Conv3D(32, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same', kernel_regularizer=regularizers.l2(0.01))(
        i)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv3D(32, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same', kernel_regularizer=regularizers.l2(0.01))(
        conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    pool1 = Dropout(0.1)(pool1)

    conv2 = Conv3D(64, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same', kernel_regularizer=regularizers.l2(0.01))(
        pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv3D(64, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same', kernel_regularizer=regularizers.l2(0.01))(
        conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    pool2 = Dropout(0.1)(pool2)

    conv3 = Conv3D(128, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv3D(128, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    pool3 = Dropout(0.1)(pool3)

    conv4 = Conv3D(256, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv3D(256, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)
    pool4 = Dropout(0.1)(pool4)

    conv5 = Conv3D(512, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.05))(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv3D(512, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.05))(conv5)
    conv5 = BatchNormalization()(conv5)


    up6 = UpSampling3D(size=(2, 2, 2))(conv5)
    up6 = Concatenate(axis=4)([up6, conv4])
    up6 = Dropout(0.1)(up6)
    conv6 = Conv3D(256, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.05))(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv3D(256, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.05))(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = UpSampling3D(size=(2, 2, 2))(conv6)
    up7 = Concatenate(axis=4)([up7, conv3])
    up7 = Dropout(0.1)(up7)
    conv7 = Conv3D(128, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv3D(128, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = UpSampling3D(size=(2, 2, 2))(conv7)
    up8 = Concatenate(axis=4)([up8, conv2])
    up8 = Dropout(0.1)(up8)
    conv8 = Conv3D(64, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same', kernel_regularizer=regularizers.l2(0.05))(
        up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv3D(64, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same', kernel_regularizer=regularizers.l2(0.01))(
        conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = UpSampling3D(size=(2, 2, 2))(conv8)
    up9 = Concatenate(axis=4)([up9, conv1])
    up9 = Dropout(0.1)(up9)
    conv9 = Conv3D(32, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same', kernel_regularizer=regularizers.l2(0.01),
                   name='last_3')(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv3D(32, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same', kernel_regularizer=regularizers.l2(0.01),
                   name='last_2')(conv9)
    conv9 = BatchNormalization()(conv9)

    pred = Conv3D(n_classes, kernel_size=(1, 1, 1), activation='softmax', padding='valid', name='last_1')(conv9)
    final_model = Model(inputs=i, outputs=pred, name=model_name)

    return final_model

def unet_3d_do_innerlayer(input_shape, model_name, n_classes, pretrained = False):
    i = Input((input_shape[0], input_shape[1], input_shape[2], input_shape[3]))
    i /= 255.
    i -= 0.5
    i *= 2.
    # s = Lambda(lambda x: preprocess_input(x))(i)

    if pretrained:
        path = os.path.join('models', model_name+'.model')
        if os.path.exists(path):
            model = load_model(path, custom_objects={'dice': dice})
            model.summary()
            return model
        else:
            print('Failed to load existing model at: {}'.format(path))

    conv1 = Conv3D(32, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same', kernel_regularizer=regularizers.l2(0.01))(
        i)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv3D(32, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same', kernel_regularizer=regularizers.l2(0.01))(
        conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    #pool1 = Dropout(0.1)(pool1)

    conv2 = Conv3D(64, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same', kernel_regularizer=regularizers.l2(0.01))(
        pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv3D(64, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same', kernel_regularizer=regularizers.l2(0.01))(
        conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    #pool2 = Dropout(0.1)(pool2)

    conv3 = Conv3D(128, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv3D(128, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    pool3 = Dropout(0.1)(pool3)

    conv4 = Conv3D(256, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv3D(256, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)
    pool4 = Dropout(0.1)(pool4)

    conv5 = Conv3D(512, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.05))(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv3D(512, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.05))(conv5)
    conv5 = BatchNormalization()(conv5)


    up6 = UpSampling3D(size=(2, 2, 2))(conv5)
    up6 = Concatenate(axis=4)([up6, conv4])
    up6 = Dropout(0.1)(up6)
    conv6 = Conv3D(256, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.05))(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv3D(256, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.05))(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = UpSampling3D(size=(2, 2, 2))(conv6)
    up7 = Concatenate(axis=4)([up7, conv3])
    up7 = Dropout(0.1)(up7)
    conv7 = Conv3D(128, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv3D(128, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = UpSampling3D(size=(2, 2, 2))(conv7)
    up8 = Concatenate(axis=4)([up8, conv2])
    #up8 = Dropout(0.1)(up8)
    conv8 = Conv3D(64, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same', kernel_regularizer=regularizers.l2(0.05))(
        up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv3D(64, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same', kernel_regularizer=regularizers.l2(0.01))(
        conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = UpSampling3D(size=(2, 2, 2))(conv8)
    up9 = Concatenate(axis=4)([up9, conv1])
    #up9 = Dropout(0.1)(up9)
    conv9 = Conv3D(32, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same', kernel_regularizer=regularizers.l2(0.01),
                   name='last_3')(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv3D(32, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same', kernel_regularizer=regularizers.l2(0.01),
                   name='last_2')(conv9)
    conv9 = BatchNormalization()(conv9)

    pred = Conv3D(n_classes, kernel_size=(1, 1, 1), activation='softmax', padding='valid', name='last_1')(conv9)
    final_model = Model(inputs=i, outputs=pred, name=model_name)

    return final_model

# 3d UNET MODEL INST NORM
def unet_3d_inst_norm(input_shape, model_name, n_classes, pretrained = False):
    i = Input((input_shape[0], input_shape[1], input_shape[2], input_shape[3]))
    i /= 255.
    i -= 0.5
    i *= 2.
    #s = Lambda(lambda x: preprocess_input(x))(i)

    if pretrained:
        path = os.path.join('models', model_name+'.model')
        if os.path.exists(path):
            model = load_model(path, custom_objects={'dice': dice})
            model.summary()
            return model
        else:
            print('Failed to load existing model at: {}'.format(path))

    conv1 = Conv3D(32, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same', kernel_regularizer=regularizers.l2(0.01))(
        i)
    conv1 = InstanceNormalization()(conv1)
    conv1 = Conv3D(32, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same', kernel_regularizer=regularizers.l2(0.01))(
        conv1)
    conv1 = InstanceNormalization()(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    #pool1 = Dropout(0.1)(pool1)

    conv2 = Conv3D(64, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same', kernel_regularizer=regularizers.l2(0.01))(
        pool1)
    conv2 = InstanceNormalization()(conv2)
    conv2 = Conv3D(64, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same', kernel_regularizer=regularizers.l2(0.01))(
        conv2)
    conv2 = InstanceNormalization()(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    #pool2 = Dropout(0.1)(pool2)

    conv3 = Conv3D(128, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(pool2)
    conv3 = InstanceNormalization()(conv3)
    conv3 = Conv3D(128, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(conv3)
    conv3 = InstanceNormalization()(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    pool3 = Dropout(0.1)(pool3)

    conv4 = Conv3D(256, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(pool3)
    conv4 = InstanceNormalization()(conv4)
    conv4 = Conv3D(256, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(conv4)
    conv4 = InstanceNormalization()(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)
    pool4 = Dropout(0.1)(pool4)

    conv5 = Conv3D(512, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(pool4)
    conv5 = InstanceNormalization()(conv5)
    conv5 = Conv3D(512, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(conv5)
    conv5 = InstanceNormalization()(conv5)


    up6 = UpSampling3D(size=(2, 2, 2))(conv5)
    up6 = Concatenate(axis=4)([up6, conv4])
    up6 = Dropout(0.1)(up6)
    conv6 = Conv3D(256, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(up6)
    conv6 = InstanceNormalization()(conv6)
    conv6 = Conv3D(256, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(conv6)
    conv6 = InstanceNormalization()(conv6)

    up7 = UpSampling3D(size=(2, 2, 2))(conv6)
    up7 = Concatenate(axis=4)([up7, conv3])
    up7 = Dropout(0.1)(up7)
    conv7 = Conv3D(128, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(up7)
    conv7 = InstanceNormalization()(conv7)
    conv7 = Conv3D(128, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(conv7)
    conv7 = InstanceNormalization()(conv7)

    up8 = UpSampling3D(size=(2, 2, 2))(conv7)
    up8 = Concatenate(axis=4)([up8, conv2])
    #up8 = Dropout(0.1)(up8)
    conv8 = Conv3D(64, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same', kernel_regularizer=regularizers.l2(0.01))(
        up8)
    conv8 = InstanceNormalization()(conv8)
    conv8 = Conv3D(64, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same', kernel_regularizer=regularizers.l2(0.01))(
        conv8)
    conv8 = InstanceNormalization()(conv8)

    up9 = UpSampling3D(size=(2, 2, 2))(conv8)
    up9 = Concatenate(axis=4)([up9, conv1])
    #up9 = Dropout(0.1)(up9)
    conv9 = Conv3D(32, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same', kernel_regularizer=regularizers.l2(0.01),
                   name='last_3')(up9)
    conv9 = InstanceNormalization()(conv9)
    conv9 = Conv3D(32, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same', kernel_regularizer=regularizers.l2(0.01),
                   name='last_2')(conv9)
    conv9 = InstanceNormalization()(conv9)

    pred = Conv3D(n_classes, kernel_size=(1, 1, 1), activation='softmax', padding='valid', name='last_1')(conv9)
    final_model = Model(inputs=i, outputs=pred, name=model_name)


    return final_model


# 3d UNET MODEL INST NORM
def unet_3d_inst_norm_elu(input_shape, model_name, n_classes, pretrained = False):
    i = Input((input_shape[0], input_shape[1], input_shape[2], input_shape[3]))
    i /= 255.
    i -= 0.5
    i *= 2.
    #s = Lambda(lambda x: preprocess_input(x))(i)

    if pretrained:
        path = os.path.join('models', model_name+'.model')
        if os.path.exists(path):
            model = load_model(path, custom_objects={'dice': dice})
            model.summary()
            return model
        else:
            print('Failed to load existing model at: {}'.format(path))

    conv1 = Conv3D(32, kernel_size=(3, 3, 3), activation='elu', padding='same', kernel_regularizer=regularizers.l2(0.01))(
        i)
    conv1 = InstanceNormalization()(conv1)
    conv1 = Conv3D(32, kernel_size=(3, 3, 3), activation='elu', padding='same', kernel_regularizer=regularizers.l2(0.01))(
        conv1)
    conv1 = InstanceNormalization()(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    pool1 = Dropout(0.1)(pool1)

    conv2 = Conv3D(64, kernel_size=(3, 3, 3), activation='elu', padding='same', kernel_regularizer=regularizers.l2(0.01))(
        pool1)
    conv2 = InstanceNormalization()(conv2)
    conv2 = Conv3D(64, kernel_size=(3, 3, 3), activation='elu', padding='same', kernel_regularizer=regularizers.l2(0.01))(
        conv2)
    conv2 = InstanceNormalization()(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    pool2 = Dropout(0.1)(pool2)

    conv3 = Conv3D(128, kernel_size=(3, 3, 3), activation='elu', padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(pool2)
    conv3 = InstanceNormalization()(conv3)
    conv3 = Conv3D(128, kernel_size=(3, 3, 3), activation='elu', padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(conv3)
    conv3 = InstanceNormalization()(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    pool3 = Dropout(0.1)(pool3)

    conv4 = Conv3D(256, kernel_size=(3, 3, 3), activation='elu', padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(pool3)
    conv4 = InstanceNormalization()(conv4)
    conv4 = Conv3D(256, kernel_size=(3, 3, 3), activation='elu', padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(conv4)
    conv4 = InstanceNormalization()(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)
    pool4 = Dropout(0.1)(pool4)

    conv5 = Conv3D(512, kernel_size=(3, 3, 3), activation='elu', padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(pool4)
    conv5 = InstanceNormalization()(conv5)
    conv5 = Conv3D(512, kernel_size=(3, 3, 3), activation='elu', padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(conv5)
    conv5 = InstanceNormalization()(conv5)


    up6 = UpSampling3D(size=(2, 2, 2))(conv5)
    up6 = Concatenate(axis=4)([up6, conv4])
    up6 = Dropout(0.1)(up6)
    conv6 = Conv3D(256, kernel_size=(3, 3, 3), activation='elu', padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(up6)
    conv6 = InstanceNormalization()(conv6)
    conv6 = Conv3D(256, kernel_size=(3, 3, 3), activation='elu', padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(conv6)
    conv6 = InstanceNormalization()(conv6)

    up7 = UpSampling3D(size=(2, 2, 2))(conv6)
    up7 = Concatenate(axis=4)([up7, conv3])
    up7 = Dropout(0.1)(up7)
    conv7 = Conv3D(128, kernel_size=(3, 3, 3), activation='elu', padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(up7)
    conv7 = InstanceNormalization()(conv7)
    conv7 = Conv3D(128, kernel_size=(3, 3, 3), activation='elu', padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(conv7)
    conv7 = InstanceNormalization()(conv7)

    up8 = UpSampling3D(size=(2, 2, 2))(conv7)
    up8 = Concatenate(axis=4)([up8, conv2])
    up8 = Dropout(0.1)(up8)
    conv8 = Conv3D(64, kernel_size=(3, 3, 3), activation='elu', padding='same', kernel_regularizer=regularizers.l2(0.01))(
        up8)
    conv8 = InstanceNormalization()(conv8)
    conv8 = Conv3D(64, kernel_size=(3, 3, 3), activation='elu', padding='same', kernel_regularizer=regularizers.l2(0.01))(
        conv8)
    conv8 = InstanceNormalization()(conv8)

    up9 = UpSampling3D(size=(2, 2, 2))(conv8)
    up9 = Concatenate(axis=4)([up9, conv1])
    up9 = Dropout(0.1)(up9)
    conv9 = Conv3D(32, kernel_size=(3, 3, 3), activation='elu', padding='same', kernel_regularizer=regularizers.l2(0.01),
                   name='last_3')(up9)
    conv9 = InstanceNormalization()(conv9)
    conv9 = Conv3D(32, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same', kernel_regularizer=regularizers.l2(0.01),
                   name='last_2')(conv9)
    conv9 = InstanceNormalization()(conv9)

    pred = Conv3D(n_classes, kernel_size=(1, 1, 1), activation='softmax', padding='valid', name='last_1')(conv9)
    final_model = Model(inputs=i, outputs=pred, name=model_name)

    return final_model


# UNET MODEL WITH ATTENTION GATE FOR DEEP SUPERVISION
def attention_unet_jaye(input_shape, model_name, n_classes, pretrained = False):
    i = Input((input_shape[0], input_shape[1], input_shape[2]))
    s = Lambda(lambda x: preprocess_input(x))(i)

    if pretrained:
        path = os.path.join('models', model_name+'.model')
        if os.path.exists(path):
            model = load_model(path, custom_objects={'dice': dice})
            model.summary()
            return model
        else:
            print('Failed to load existing model at: {}'.format(path))

    conv1 = Conv2D(32, kernel_size=(3, 3), activation=LeakyReLU(), padding='same', kernel_regularizer=regularizers.l2(0.01))(
        s)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, kernel_size=(3, 3), activation=LeakyReLU(), padding='same', kernel_regularizer=regularizers.l2(0.01))(
        conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(0.1)(pool1)

    conv2 = Conv2D(64, kernel_size=(3, 3), activation=LeakyReLU(), padding='same', kernel_regularizer=regularizers.l2(0.01))(
        pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, kernel_size=(3, 3), activation=LeakyReLU(), padding='same', kernel_regularizer=regularizers.l2(0.01))(
        conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(0.1)(pool2)

    conv3 = Conv2D(128, kernel_size=(3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, kernel_size=(3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(0.1)(pool3)

    conv4 = Conv2D(256, kernel_size=(3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, kernel_size=(3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = Dropout(0.1)(pool4)

    conv5 = Conv2D(512, kernel_size=(3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(512, kernel_size=(3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(conv5)
    conv5 = BatchNormalization()(conv5)


    up6 = UpSampling2D(size=(2, 2))(conv5)
    up6 = Concatenate(axis=3)([up6, conv4])
    up6 = attention_block(up6, conv4, 256, False)
    up6 = Dropout(0.1)(up6)
    conv6 = Conv2D(256, kernel_size=(3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(256, kernel_size=(3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = Concatenate(axis=3)([up7, conv3])
    up7 = attention_block(up7, conv3, 128, False)

    up7 = Dropout(0.1)(up7)
    conv7 = Conv2D(128, kernel_size=(3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(128, kernel_size=(3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(conv7)
    conv7 = BatchNormalization()(conv7)



    up8 = UpSampling2D(size=(2, 2))(conv7)
    up8 = Concatenate(axis=3)([up8, conv2])
    up8 = attention_block(up8, conv2, 64, False)
    up8 = Dropout(0.1)(up8)
    conv8 = Conv2D(64, kernel_size=(3, 3), activation=LeakyReLU(), padding='same', kernel_regularizer=regularizers.l2(0.01))(
        up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(64, kernel_size=(3, 3), activation=LeakyReLU(), padding='same', kernel_regularizer=regularizers.l2(0.01))(
        conv8)
    conv8 = BatchNormalization()(conv8)


    up9 = UpSampling2D(size=(2, 2))(conv8)
    up9 = Concatenate(axis=3)([up9, conv1])
    up9 = attention_block(up9, conv1, 32, False)
    up9 = Dropout(0.1)(up9)
    conv9 = Conv2D(32, kernel_size=(3, 3), activation=LeakyReLU(), padding='same', kernel_regularizer=regularizers.l2(0.01),
                   name='last_3')(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(32, kernel_size=(3, 3), activation=LeakyReLU(), padding='same', kernel_regularizer=regularizers.l2(0.01),
                   name='last_2')(conv9)
    conv9 = BatchNormalization()(conv9)



    pred = Conv2D(n_classes, kernel_size=(1, 1), activation='softmax', padding='valid', name='last_1')(conv9)
    final_model = Model(inputs=i, outputs=pred, name=model_name)

    return final_model


def attention_block(F_g, F_l, F_int, bn=False):
    g = Conv2D(F_int, kernel_size=(1, 1), strides=(1, 1), padding='valid')(F_g)
    if bn:
        g = BatchNormalization()(g)
    x = Conv2D(F_int, kernel_size=(1, 1), strides=(1, 1), padding='valid')(F_l)
    if bn:
        x = BatchNormalization()(x)

    psi = Add()([g, x])
    psi = Activation('relu')(psi)

    psi = Conv2D(1, kernel_size=(1, 1), strides=(1, 1), padding='valid')(psi)

    if bn:
        psi = BatchNormalization()(psi)
    psi = Activation('sigmoid')(psi)

    return Multiply()([F_l, psi])

# 3d - BATCH NORM - GAUSSIAN DROPOUT
def unet_3d_gaussian_do(input_shape, model_name, n_classes, pretrained=False):
    i = Input((input_shape[0], input_shape[1], input_shape[2], input_shape[3]))
    i /= 255.
    i -= 0.5
    i *= 2.

    if pretrained:
        path = os.path.join('models', model_name + '.model')
        if os.path.exists(path):
            model = load_model(path, custom_objects={'dice': dice})
            model.summary()
            return model
        else:
            print('Failed to load existing model at: {}'.format(path))

    conv1 = Conv3D(32, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(
        i)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv3D(32, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(
        conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    pool1 = GaussianDropout(0.1)(pool1)

    conv2 = Conv3D(64, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(
        pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv3D(64, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(
        conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    pool2 = GaussianDropout(0.1)(pool2)

    conv3 = Conv3D(128, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv3D(128, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    pool3 = GaussianDropout(0.1)(pool3)

    conv4 = Conv3D(256, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv3D(256, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)
    pool4 = GaussianDropout(0.1)(pool4)

    conv5 = Conv3D(512, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv3D(512, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(conv5)
    conv5 = BatchNormalization()(conv5)

    up6 = UpSampling3D(size=(2, 2, 2))(conv5)
    up6 = Concatenate(axis=4)([up6, conv4])
    up6 = GaussianDropout(0.1)(up6)
    conv6 = Conv3D(256, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv3D(256, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = UpSampling3D(size=(2, 2, 2))(conv6)
    up7 = Concatenate(axis=4)([up7, conv3])
    up7 = GaussianDropout(0.1)(up7)
    conv7 = Conv3D(128, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv3D(128, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = UpSampling3D(size=(2, 2, 2))(conv7)
    up8 = Concatenate(axis=4)([up8, conv2])
    up8 = GaussianDropout(0.1)(up8)
    conv8 = Conv3D(64, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(
        up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv3D(64, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(
        conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = UpSampling3D(size=(2, 2, 2))(conv8)
    up9 = Concatenate(axis=4)([up9, conv1])
    up9 = GaussianDropout(0.1)(up9)
    conv9 = Conv3D(32, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01),
                   name='last_3')(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv3D(32, kernel_size=(3, 3, 3), activation=LeakyReLU(), padding='same',
                   kernel_regularizer=regularizers.l2(0.01),
                   name='last_2')(conv9)
    conv9 = BatchNormalization()(conv9)

    pred = Conv3D(n_classes, kernel_size=(1, 1, 1), activation='softmax', padding='valid', name='last_1')(conv9)
    final_model = Model(inputs=i, outputs=pred, name=model_name)

    return final_model



