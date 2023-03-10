from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from attention_blocks import *

"""
This code is inspired by arkanivasarkar where he uses u-nets for retinal vessel segmentation:
https://github.com/arkanivasarkar/Retinal-Vessel-Segmentation-using-variants-of-UNET

The code has been re-written and adapted for my project and for brain tumor segmentation
"""

def build_attention_unet(n_channels, ker_init, dropout, batchnorm=True):

    inputs = Input((128, 128, n_channels))

    # Downsampling layers
    conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=ker_init)(inputs)
    conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=ker_init)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=ker_init)(pool1)
    conv2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=ker_init)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=ker_init)(pool2)
    conv3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=ker_init)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=ker_init)(pool3)
    conv4 = Dropout(dropout)(conv4)
    conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=ker_init)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=ker_init)(pool4)
    conv5 = Dropout(dropout)(conv5)
    conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=ker_init)(conv5)


    # Upsampling layers
    gating_5 = gatingsignal(conv5, 128, batchnorm)
    att_5 = attention_block(conv4, gating_5, 128)
    up_5 = UpSampling2D(size=(2, 2))(conv5)
    up_5 = concatenate([up_5, att_5], axis=3)
    up_conv_5 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=ker_init)(up_5)
    drop6 = Dropout(dropout)(up_conv_5)
    up_conv_5 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=ker_init)(drop6)

    gating_4 = gatingsignal(up_conv_5, 64, batchnorm)
    att_4 = attention_block(conv3, gating_4, 64)
    up_4 = UpSampling2D(size=(2, 2))(up_conv_5)
    up_4 = concatenate([up_4, att_4], axis=3)
    up_conv_4 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=ker_init)(up_4)
    drop7 = Dropout(dropout)(up_conv_4)
    up_conv_4 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=ker_init)(drop7)

    gating_3 = gatingsignal(up_conv_4, 32, batchnorm)
    att_3 = attention_block(conv2, gating_3, 32)
    up_3 = UpSampling2D(size=(2, 2))(up_conv_4)
    up_3 = concatenate([up_3, att_3], axis=3)
    up_conv_3 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=ker_init)(up_3)
    drop8 = Dropout(dropout)(up_conv_3)
    up_conv_3 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=ker_init)(drop8)

    gating_2 = gatingsignal(up_conv_3, 16, batchnorm)
    att_2 = attention_block(conv1, gating_2, 16)
    up_2 = UpSampling2D(size=(2, 2))(up_conv_3)
    up_2 = concatenate([up_2, att_2], axis=3)
    up_conv_2 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer=ker_init)(up_2)
    drop9 = Dropout(dropout)(up_conv_2)
    up_conv_2 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer=ker_init)(drop9)

    outputs = Conv2D(4, (1, 1), activation='softmax')(up_conv_2)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.summary()
    return model