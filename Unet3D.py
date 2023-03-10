from tensorflow.keras.layers import *
from tensorflow.keras.models import *

"""
This code is inspired by arkanivasarkar where he uses u-nets for retinal vessel segmentation:
https://github.com/arkanivasarkar/Retinal-Vessel-Segmentation-using-variants-of-UNET

The code has been re-written to a 3D U-Net and modeified for my project
"""


def build_3Dunet(n_channels, ker_init, dropout):
    inputs = Input((128, 128, 128, n_channels))

    conv1 = Conv3D(16, (3, 3, 3), activation='relu', padding='same', kernel_initializer=ker_init)(inputs)
    conv1 = Conv3D(16, (3, 3, 3), activation='relu', padding='same', kernel_initializer=ker_init)(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer=ker_init)(pool1)
    conv2 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer=ker_init)(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer=ker_init)(pool2)
    conv3 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer=ker_init)(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_initializer=ker_init)(pool3)
    conv4 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_initializer=ker_init)(conv4)
    drop4 = Dropout(dropout)(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(drop4)

    conv5 = Conv3D(256, (3, 3, 3), activation='relu', padding='same', kernel_initializer=ker_init)(pool4)
    conv5 = Conv3D(256, (3, 3, 3), activation='relu', padding='same', kernel_initializer=ker_init)(conv5)
    drop5 = Dropout(dropout)(conv5)

    up6 = Conv3D(128, (2, 2, 2), activation='relu', padding='same', kernel_initializer=ker_init)(
        UpSampling3D(size=(2, 2, 2))(drop5))
    merge6 = concatenate([drop4, up6])
    conv6 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_initializer=ker_init)(merge6)
    conv6 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_initializer=ker_init)(conv6)

    up7 = Conv3D(64, (2, 2, 2), activation='relu', padding='same', kernel_initializer=ker_init)(
        UpSampling3D(size=(2, 2, 2))(conv6))
    merge7 = concatenate([conv3, up7])
    conv7 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer=ker_init)(merge7)
    conv7 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer=ker_init)(conv7)

    up8 = Conv3D(32, (2, 2, 2), activation='relu', padding='same', kernel_initializer=ker_init)(
        UpSampling3D(size=(2, 2, 2))(conv7))
    merge8 = concatenate([conv2, up8])
    conv8 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer=ker_init)(merge8)
    conv8 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer=ker_init)(conv8)

    up9 = Conv3D(16, (2, 2, 2), activation='relu', padding='same', kernel_initializer=ker_init)(
        UpSampling3D(size=(2, 2, 2))(conv8))
    merge9 = concatenate([conv1, up9])
    conv9 = Conv3D(16, (3, 3, 3), activation='relu', padding='same', kernel_initializer=ker_init)(merge9)
    conv9 = Conv3D(16, (3, 3, 3), activation='relu', padding='same', kernel_initializer=ker_init)(conv9)

    outputs = Conv2D(4, (1, 1), activation='softmax')(conv9)
    model = Model(inputs=[inputs], outputs=[outputs])
    model.summary()
    return model