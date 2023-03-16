from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import keras.backend as K


"""
This code is originally written by arkanivasarkar where he uses u-nets for retinal vessel segmentation:
https://github.com/arkanivasarkar/Retinal-Vessel-Segmentation-using-variants-of-UNET
This code has been adapted to my project.

Changes made to the code include:
* Softmax implemented instead of sigmoid
* Activation function is experimented with, e.g., LeakyRelu implemented instead of relu.
* Batch normalization is experimented with and without. 
"""
def gatingsignal(input, out_size, batchnorm=False):
    x = Conv2D(out_size, (1, 1), padding='same')(input)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def attention_block(x, gating, inter_shape):
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(gating)
    theta_x = Conv2D(inter_shape, (2, 2), strides=(2, 2), kernel_initializer='he_normal', padding='same')(x)
    shape_theta_x = K.int_shape(theta_x)
    phi_g = Conv2D(inter_shape, (1, 1), kernel_initializer='he_normal', padding='same')(gating)
    upsample_g = Conv2DTranspose(inter_shape, (3, 3), strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]), kernel_initializer='he_normal', padding='same')(phi_g)
    concat_xg = add([upsample_g, theta_x])
    act_xg = Activation('relu')(concat_xg)
    psi = Conv2D(1, (1, 1), kernel_initializer='he_normal', padding='same')(act_xg)
    softmax_xg = Activation('softmax')(psi)
    shape_softmax = K.int_shape(softmax_xg)
    upsample_psi = UpSampling2D(size=(shape_x[1] // shape_softmax[1], shape_x[2] // shape_softmax[2]))(softmax_xg)
    upsample_psi = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': shape_x[3]})(upsample_psi)
    y = multiply([upsample_psi, x])
    result = Conv2D(shape_x[3], (1, 1), kernel_initializer='he_normal', padding='same')(y)
    attenblock = BatchNormalization()(result)
    return attenblock



"""
This code is inspired by arkanivasarkar where he uses u-nets for retinal vessel segmentation:
https://github.com/arkanivasarkar/Retinal-Vessel-Segmentation-using-variants-of-UNET

The code has been re-written and adapted for my project and for brain tumor segmentation. 
Changes made in the models trained:

* Softmax implemented instead of sigmoid in the final layer. 
* Dropout is experimented with in by adding/removing dropout in every block
* Batch-normalization is experimented with in by adding/removing Batch-normalization in every block
* Number of channels used are changed
* Activation function used is experimented with in every convolutional layer. e.g., LeakyRelu implemented instead of relu. 
* The overall structure of the code is changed
"""


def build_resatt_unet(n_channels, ker_init, dropout=0, batchnorm=False):
    inputs = Input((128, 128, n_channels))

    conv1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer=ker_init, padding='same')(inputs)
    conv2 = Conv2D(16, (3, 3), activation='relu', kernel_initializer=ker_init, padding='same')(conv1)
    conv2 = Dropout(dropout)(conv2)
    shortcut1 = Conv2D(16, (1, 1), activation='relu', kernel_initializer=ker_init, padding='same')(inputs)
    residual_path1 = add([shortcut1, conv2])
    pool1 = MaxPooling2D(pool_size=(2, 2))(residual_path1)

    conv3 = Conv2D(32, (3, 3), activation='relu', kernel_initializer=ker_init, padding='same')(pool1)
    conv4 = Conv2D(32, (3, 3), activation='relu', kernel_initializer=ker_init, padding='same')(conv3)
    conv4 = Dropout(dropout)(conv4)
    shortcut2 = Conv2D(32, (1, 1), activation='relu', kernel_initializer=ker_init, padding='same')(pool1)
    residual_path2 = add([shortcut2, conv4])
    pool2 = MaxPooling2D(pool_size=(2, 2))(residual_path2)

    conv5 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=ker_init, padding='same')(pool2)
    conv6 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=ker_init, padding='same')(conv5)
    conv6 = Dropout(dropout)(conv6)
    shortcut3 = Conv2D(64, (1, 1), activation='relu', kernel_initializer=ker_init, padding='same')(pool2)
    residual_path3 = add([shortcut3, conv6])
    pool3 = MaxPooling2D(pool_size=(2, 2))(residual_path3)

    conv7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=ker_init, padding='same')(pool3)
    conv8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=ker_init, padding='same')(conv7)
    conv8 = Dropout(dropout)(conv8)
    shortcut4 = Conv2D(128, (1, 1), activation='relu', kernel_initializer=ker_init, padding='same')(pool3)
    residual_path4 = add([shortcut4, conv8])
    pool4 = MaxPooling2D(pool_size=(2, 2))(residual_path4)

    conv9 = Conv2D(256, (3, 3), activation='relu', kernel_initializer=ker_init, padding='same')(pool4)
    conv10 = Conv2D(256, (3, 3), activation='relu', kernel_initializer=ker_init, padding='same')(conv9)
    conv10 = Dropout(dropout)(conv10)
    shortcut5 = Conv2D(256, (1, 1), activation='relu', kernel_initializer=ker_init, padding='same')(pool4)
    residual_path5 = add([shortcut5, conv10])

    gating_5 = gatingsignal(residual_path5, 128, batchnorm)
    att_5 = attention_block(residual_path4, gating_5, 128)
    up_5 = UpSampling2D(size=(2, 2))(residual_path5)
    up_5 = concatenate([up_5, att_5], axis=3)
    conv11 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=ker_init, padding='same')(up_5)
    conv12 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=ker_init, padding='same')(conv11)
    conv12 = Dropout(dropout)(conv12)
    shortcut6 = Conv2D(128, (1, 1), activation='relu', kernel_initializer=ker_init, padding='same')(up_5)
    residual_path6 = add([shortcut6, conv12])

    gating_4 = gatingsignal(residual_path6, 64, batchnorm)
    att_4 = attention_block(residual_path3, gating_4, 64)
    up_4 = UpSampling2D(size=(2, 2))(residual_path6)
    up_4 = concatenate([up_4, att_4], axis=3)
    conv13 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=ker_init, padding='same')(up_4)
    conv14 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=ker_init, padding='same')(conv13)
    conv14 = Dropout(dropout)(conv14)
    shortcut7 = Conv2D(64, (1, 1), activation='relu', kernel_initializer=ker_init, padding='same')(up_4)
    residual_path7 = add([shortcut7, conv14])

    gating_3 = gatingsignal(residual_path7, 32, batchnorm)
    att_3 = attention_block(residual_path2, gating_3, 32)
    up_3 = UpSampling2D(size=(2, 2))(residual_path7)
    up_3 = concatenate([up_3, att_3], axis=3)
    conv15 = Conv2D(32, (3, 3), activation='relu', kernel_initializer=ker_init, padding='same')(up_3)
    conv16 = Conv2D(32, (3, 3), activation='relu', kernel_initializer=ker_init, padding='same')(conv15)
    conv16 = Dropout(dropout)(conv16)
    shortcut8 = Conv2D(32, (1, 1), activation='relu', kernel_initializer=ker_init, padding='same')(up_3)
    residual_path8 = add([shortcut8, conv16])

    gating_2 = gatingsignal(residual_path8, 16, batchnorm)
    att_2 = attention_block(residual_path1, gating_2, 16)
    up_2 = UpSampling2D(size=(2, 2))(residual_path8)
    up_2 = concatenate([up_2, att_2], axis=3)
    conv17 = Conv2D(16, (3, 3), activation='relu', kernel_initializer=ker_init, padding='same')(up_2)
    conv18 = Conv2D(16, (3, 3), activation='relu', kernel_initializer=ker_init, padding='same')(conv17)
    conv18 = Dropout(dropout)(conv18)
    shortcut9 = Conv2D(16, (1, 1), activation='relu', kernel_initializer=ker_init, padding='same')(up_2)
    residual_path9 = add([shortcut9, conv18])

    outputs = Conv2D(4, (1, 1), activation='softmax')(residual_path9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.summary()
    return model