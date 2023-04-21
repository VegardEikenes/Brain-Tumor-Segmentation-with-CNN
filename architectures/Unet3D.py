from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow_addons.layers import *

"""
This code is inspired by Naomi Fridman: https://naomi-fridman.medium.com/multi-class-image-segmentation-a5cc671e647a and
has been adapted to my project. 

The code was originally a base 2D U-Net. The code has been rewritten for my project.
Changes made to the code:
* 2D convolutions replaced by 3D convolutions.
* 2D max-pooling replaced by 3D max-pooling.
* Number of filters used in each layer is reduced. 
* Kernel-sizes changed from 3,3 to 3,3,3 in convolutional layers (added dimension)
* Pool-size of max-pooling changed from 2,2 to 2,2,2 in max-pooling layers (added dimension)
* UpSampling size changed from 2,2 to 2,2,2 (added dimension)
* Activation functions are experimented with in every conv layer for the trained models, e.g., LeakyRelu implemented instead of Relu
* Dropout is experimented with for by adding/removing dropout in every block
* Batch normalization and instance normalization is experimented with by adding/removing it in every block
"""


def build_3Dunet(n_channels, ker_init, dropout):
    inputs = Input((128, 128, 128, n_channels))

    conv1 = Conv3D(16, (3, 3, 3), padding='same', kernel_initializer=ker_init)(inputs)
    conv1 = InstanceNormalization(axis=-1)(conv1)
    conv1 = Activation("relu")(conv1)
    conv1 = Dropout(dropout)(conv1)
    conv1 = Conv3D(16, (3, 3, 3), padding='same', kernel_initializer=ker_init)(conv1)
    conv1 = InstanceNormalization(axis=-1)(conv1)
    conv1 = Activation("relu")(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(32, (3, 3, 3), padding='same', kernel_initializer=ker_init)(pool1)
    conv2 = InstanceNormalization(axis=-1)(conv2)
    conv2 = Activation("relu")(conv2)
    conv2 = Dropout(dropout)(conv2)
    conv2 = Conv3D(32, (3, 3, 3), padding='same', kernel_initializer=ker_init)(conv2)
    conv2 = InstanceNormalization(axis=-1)(conv2)
    conv2 = Activation("relu")(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(64, (3, 3, 3), padding='same', kernel_initializer=ker_init)(pool2)
    conv3 = InstanceNormalization(axis=-1)(conv3)
    conv3 = Activation("relu")(conv3)
    conv3 = Dropout(dropout)(conv3)
    conv3 = Conv3D(64, (3, 3, 3), padding='same', kernel_initializer=ker_init)(conv3)
    conv3 = InstanceNormalization(axis=-1)(conv3)
    conv3 = Activation("relu")(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = Conv3D(128, (3, 3, 3), padding='same', kernel_initializer=ker_init)(pool3)
    conv4 = InstanceNormalization(axis=-1)(conv4)
    conv4 = Activation("relu")(conv4)
    conv4 = Dropout(dropout)(conv4)
    conv4 = Conv3D(128, (3, 3, 3), padding='same', kernel_initializer=ker_init)(conv4)
    conv4 = InstanceNormalization(axis=-1)(conv4)
    conv4 = Activation("relu")(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

    conv5 = Conv3D(256, (3, 3, 3), padding='same', kernel_initializer=ker_init)(pool4)
    conv5 = InstanceNormalization(axis=-1)(conv5)
    conv5 = Activation("relu")(conv5)
    conv5 = Dropout(dropout)(conv5)
    conv5 = Conv3D(256, (3, 3, 3), padding='same', kernel_initializer=ker_init)(conv5)
    conv5 = InstanceNormalization(axis=-1)(conv5)
    conv5 = Activation("relu")(conv5)

    up6 = Conv3D(128, (2, 2, 2), padding='same', kernel_initializer=ker_init)(
        UpSampling3D(size=(2, 2, 2))(conv5))
    merge6 = concatenate([conv4, up6])
    conv6 = Conv3D(128, (3, 3, 3), padding='same', kernel_initializer=ker_init)(merge6)
    conv6 = InstanceNormalization(axis=-1)(conv6)
    conv6 = Activation("relu")(conv6)
    conv6 = Dropout(dropout)(conv6)
    conv6 = Conv3D(128, (3, 3, 3), padding='same', kernel_initializer=ker_init)(conv6)
    conv6 = InstanceNormalization(axis=-1)(conv6)
    conv6 = Activation("relu")(conv6)

    up7 = Conv3D(64, (2, 2, 2), padding='same', kernel_initializer=ker_init)(
        UpSampling3D(size=(2, 2, 2))(conv6))
    merge7 = concatenate([conv3, up7])
    conv7 = Conv3D(64, (3, 3, 3), padding='same', kernel_initializer=ker_init)(merge7)
    conv7 = InstanceNormalization(axis=-1)(conv7)
    conv7 = Activation("relu")(conv7)
    conv7 = Dropout(dropout)(conv7)
    conv7 = Conv3D(64, (3, 3, 3), padding='same', kernel_initializer=ker_init)(conv7)
    conv7 = InstanceNormalization(axis=-1)(conv7)
    conv7 = Activation("relu")(conv7)

    up8 = Conv3D(32, (2, 2, 2), padding='same', kernel_initializer=ker_init)(
        UpSampling3D(size=(2, 2, 2))(conv7))
    merge8 = concatenate([conv2, up8])
    conv8 = Conv3D(32, (3, 3, 3), padding='same', kernel_initializer=ker_init)(merge8)
    conv8 = InstanceNormalization(axis=-1)(conv8)
    conv8 = Activation("relu")(conv8)
    conv8 = Dropout(dropout)(conv8)
    conv8 = Conv3D(32, (3, 3, 3), padding='same', kernel_initializer=ker_init)(conv8)
    conv8 = InstanceNormalization(axis=-1)(conv8)
    conv8 = Activation("relu")(conv8)

    up9 = Conv3D(16, (2, 2, 2), padding='same', kernel_initializer=ker_init)(
        UpSampling3D(size=(2, 2, 2))(conv8))
    merge9 = concatenate([conv1, up9])
    conv9 = Conv3D(16, (3, 3, 3), padding='same', kernel_initializer=ker_init)(merge9)
    conv9 = InstanceNormalization(axis=-1)(conv9)
    conv9 = Activation("relu")(conv9)
    conv9 = Dropout(dropout)(conv9)
    conv9 = Conv3D(16, (3, 3, 3), padding='same', kernel_initializer=ker_init)(conv9)
    conv9 = InstanceNormalization(axis=-1)(conv9)
    conv9 = Activation("relu")(conv9)

    outputs = Conv2D(4, (1, 1), activation='softmax')(conv9)
    model = Model(inputs=[inputs], outputs=[outputs])
    model.summary()
    return model