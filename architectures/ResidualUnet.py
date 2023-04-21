from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow_addons.layers import *
import tensorflow as tf

"""
This code is inspired by arkanivasarkar where he uses u-nets for retinal vessel segmentation:
https://github.com/arkanivasarkar/Retinal-Vessel-Segmentation-using-variants-of-UNET

Changes made in the models trained:

* Softmax implemented instead of sigmoid in the final layer. 
* Dropout is experimented with by adding/removing Dropout in every block
* Batch-normalization is experimented with by adding/removing batch normalization in every block
* Number of channels used are changed
* Activation function used is experimented with in every conv layer. e.g., LeakyRelu implemented instead of relu. 
* The general structure of the code is changed
"""
def build_res_unet(n_channels, ker_init):
    inputs = Input((128, 128, n_channels))

    conv = Conv2D(16, (3, 3), kernel_initializer=ker_init, padding="same")(inputs)
    conv = InstanceNormalization(axis=-1)(conv)
    conv = Activation("relu")(conv)
    conv1 = Conv2D(16, (3, 3), kernel_initializer=ker_init, padding="same")(conv)
    conv1 = InstanceNormalization(axis=-1)(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(32, (3, 3), kernel_initializer=ker_init, padding='same')(pool1)
    conv2 = InstanceNormalization(axis=-1)(conv2)
    conv2 = Activation('relu')(conv2)
    conv3 = Conv2D(32, (3, 3), kernel_initializer=ker_init, padding='same')(conv2)
    conv3 = InstanceNormalization(axis=-1)(conv3)
    conv3 = Activation('relu')(conv3)
    # conv3 = Dropout(0.2)(conv3)
    shortcut1 = Conv2D(32, (1, 1), kernel_initializer=ker_init, padding='same')(pool1)
    shortcut1 = InstanceNormalization(axis=-1)(shortcut1)
    shortcut1 = Activation('relu')(shortcut1)
    residual_path1 = add([shortcut1, conv3])
    pool2 = MaxPooling2D(pool_size=(2, 2))(residual_path1)

    conv4 = Conv2D(64, (3, 3), kernel_initializer=ker_init, padding='same')(pool2)
    conv4 = InstanceNormalization(axis=-1)(conv4)
    conv4 = Activation('relu')(conv4)
    conv5 = Conv2D(64, (3, 3), kernel_initializer=ker_init, padding='same')(conv4)
    conv5 = InstanceNormalization(axis=-1)(conv5)
    conv5 = Activation('relu')(conv5)
    # conv5 = Dropout(0.2)(conv5)
    shortcut2 = Conv2D(64, (1, 1), kernel_initializer=ker_init, padding='same')(pool2)
    shortcut2 = InstanceNormalization(axis=-1)(shortcut2)
    shortcut2 = Activation('relu')(shortcut2)
    residual_path2 = add([shortcut2, conv5])
    pool3 = MaxPooling2D(pool_size=(2, 2))(residual_path2)

    conv6 = Conv2D(128, (3, 3), kernel_initializer=ker_init, padding='same')(pool3)
    conv6 = InstanceNormalization(axis=-1)(conv6)
    conv6 = Activation('relu')(conv6)
    conv7 = Conv2D(128, (3, 3), kernel_initializer=ker_init, padding='same')(conv6)
    conv7 = InstanceNormalization(axis=-1)(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Dropout(0.2)(conv7)
    shortcut3 = Conv2D(128, (1, 1), kernel_initializer=ker_init, padding='same')(pool3)
    shortcut3 = InstanceNormalization(axis=-1)(shortcut3)
    shortcut3 = Activation('relu')(shortcut3)
    residual_path3 = add([shortcut3, conv7])
    pool4 = MaxPooling2D(pool_size=(2, 2))(residual_path3)

    conv8 = Conv2D(256, (3, 3), kernel_initializer=ker_init, padding='same')(pool4)
    conv8 = InstanceNormalization(axis=-1)(conv8)
    conv8 = Activation('relu')(conv8)
    conv9 = Conv2D(256, (3, 3), kernel_initializer=ker_init, padding='same')(conv8)
    conv9 = InstanceNormalization(axis=-1)(conv9)
    conv9 = Activation('relu')(conv9)
    conv9 = Dropout(0.2)(conv9)
    shortcut4 = Conv2D(256, (1, 1), kernel_initializer=ker_init, padding='same')(pool4)
    shortcut4 = Activation('relu')(shortcut4)
    shortcut4 = InstanceNormalization(axis=-1)(shortcut4)
    residual_path4 = add([shortcut4, conv9])

    up_conv1 = UpSampling2D(size=(2, 2))(residual_path4)
    up_conv1 = concatenate([up_conv1, residual_path3], axis=3)
    up_conv2 = Conv2D(128, (3, 3), kernel_initializer=ker_init, padding='same')(up_conv1)
    up_conv2 = InstanceNormalization(axis=-1)(up_conv2)
    up_conv2 = Activation('relu')(up_conv2)
    up_conv3 = Conv2D(128, (3, 3), kernel_initializer=ker_init, padding='same')(up_conv2)
    up_conv3 = InstanceNormalization(axis=-1)(up_conv3)
    up_conv3 = Activation('relu')(up_conv3)
    # up_conv3 = Dropout(0.2)(up_conv3)
    shortcut5 = Conv2D(128, (1, 1), kernel_initializer=ker_init, padding='same')(up_conv1)
    shortcut5 = InstanceNormalization(axis=-1)(shortcut5)
    shortcut5 = Activation('relu')(shortcut5)
    residual_path5 = add([shortcut5, up_conv3])

    up_conv4 = UpSampling2D(size=(2, 2))(residual_path5)
    up_conv4 = concatenate([up_conv4, residual_path2], axis=3)
    up_conv5 = Conv2D(64, (3, 3), kernel_initializer=ker_init, padding='same')(up_conv4)
    up_conv5 = InstanceNormalization(axis=-1)(up_conv5)
    up_conv5 = Activation('relu')(up_conv5)
    up_conv6 = Conv2D(64, (3, 3), kernel_initializer=ker_init, padding='same')(up_conv5)
    up_conv6 = InstanceNormalization(axis=-1)(up_conv6)
    up_conv6 = Activation('relu')(up_conv6)
    # up_conv6 = Dropout(0.2)(up_conv6)
    shortcut6 = Conv2D(64, (1, 1), kernel_initializer=ker_init, padding='same')(up_conv4)
    shortcut6 = InstanceNormalization(axis=-1)(shortcut6)
    shortcut6 = Activation('relu')(shortcut6)
    residual_path6 = add([shortcut6, up_conv6])

    up_conv7 = UpSampling2D(size=(2, 2))(residual_path6)
    up_conv7 = concatenate([up_conv7, residual_path1], axis=3)
    up_conv8 = Conv2D(32, (3, 3), kernel_initializer=ker_init, padding='same')(up_conv7)
    up_conv8 = InstanceNormalization(axis=-1)(up_conv8)
    up_conv8 = Activation('relu')(up_conv8)
    up_conv9 = Conv2D(32, (3, 3), kernel_initializer=ker_init, padding='same')(up_conv8)
    up_conv9 = InstanceNormalization(axis=-1)(up_conv9)
    up_conv9 = Activation('relu')(up_conv9)
    # up_conv9 = Dropout(0.2)(up_conv9)
    shortcut7 = Conv2D(32, (1, 1), kernel_initializer=ker_init, padding='same')(up_conv7)
    shortcut7 = InstanceNormalization(axis=-1)(shortcut7)
    shortcut7 = Activation('relu')(shortcut7)
    residual_path7 = add([shortcut7, up_conv9])

    up_conv10 = UpSampling2D(size=(2, 2))(residual_path7)
    up_conv10 = concatenate([up_conv10, conv1], axis=3)
    up_conv11 = Conv2D(16, (3, 3), kernel_initializer=ker_init, padding='same')(up_conv10)
    up_conv11 = InstanceNormalization(axis=-1)(up_conv11)
    up_conv11 = Activation('relu')(up_conv11)
    up_conv12 = Conv2D(16, (3, 3), kernel_initializer=ker_init, padding='same')(up_conv11)
    up_conv12 = InstanceNormalization(axis=-1)(up_conv12)
    up_conv12 = Activation('relu')(up_conv12)
    # up_conv12 = Dropout(0.2)(up_conv12)
    shortcut8 = Conv2D(16, (1, 1), kernel_initializer=ker_init, padding='same')(up_conv10)
    shortcut8 = InstanceNormalization(axis=-1)(shortcut8)
    shortcut8 = Activation('relu')(shortcut8)
    residual_path8 = add([shortcut8, up_conv12])

    outputs = Conv2D(4, (1, 1), activation='softmax')(residual_path8)
    model = Model(inputs=[inputs], outputs=[outputs])
    model.summary()
    return model