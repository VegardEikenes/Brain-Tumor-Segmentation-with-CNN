from tensorflow.keras.layers import *
from tensorflow.keras.models import *

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

    conv1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer=ker_init, padding="same")(inputs)
    conv1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer=ker_init, padding="same")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer=ker_init, padding='same')(pool1)
    conv3 = Conv2D(32, (3, 3), activation='relu', kernel_initializer=ker_init, padding='same')(conv2)
    shortcut1 = Conv2D(32, (1, 1), activation='relu', kernel_initializer=ker_init, padding='same')(pool1)
    residual_path1 = add([shortcut1, conv3])
    pool2 = MaxPooling2D(pool_size=(2, 2))(residual_path1)

    conv4 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=ker_init, padding='same')(pool2)
    conv5 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=ker_init, padding='same')(conv4)
    shortcut2 = Conv2D(64, (1, 1), activation='relu', kernel_initializer=ker_init, padding='same')(pool2)
    residual_path2 = add([shortcut2, conv5])
    pool3 = MaxPooling2D(pool_size=(2, 2))(residual_path2)

    conv6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=ker_init, padding='same')(pool3)
    conv7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=ker_init, padding='same')(conv6)
    shortcut3 = Conv2D(128, (1, 1), activation='relu', kernel_initializer=ker_init, padding='same')(pool3)
    residual_path3 = add([shortcut3, conv7])
    pool4 = MaxPooling2D(pool_size=(2, 2))(residual_path3)

    conv8 = Conv2D(256, (3, 3), activation='relu', kernel_initializer=ker_init, padding='same')(pool4)
    conv9 = Conv2D(256, (3, 3), activation='relu', kernel_initializer=ker_init, padding='same')(conv8)
    shortcut4 = Conv2D(256, (1, 1), activation='relu', kernel_initializer=ker_init, padding='same')(pool4)
    residual_path4 = add([shortcut4, conv9])

    up_conv1 = UpSampling2D(size=(2, 2))(residual_path4)
    up_conv1 = concatenate([up_conv1, residual_path3], axis=3)
    up_conv2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=ker_init, padding='same')(up_conv1)
    up_conv3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=ker_init, padding='same')(up_conv2)
    shortcut5 = Conv2D(128, (1, 1), activation='relu', kernel_initializer=ker_init, padding='same')(up_conv1)
    residual_path5 = add([shortcut5, up_conv3])

    up_conv4 = UpSampling2D(size=(2, 2))(residual_path5)
    up_conv4 = concatenate([up_conv4, residual_path2], axis=3)
    up_conv5 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=ker_init, padding='same')(up_conv4)
    up_conv6 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=ker_init, padding='same')(up_conv5)
    shortcut6 = Conv2D(64, (1, 1), activation='relu', kernel_initializer=ker_init, padding='same')(up_conv4)
    residual_path6 = add([shortcut6, up_conv6])

    up_conv7 = UpSampling2D(size=(2, 2))(residual_path6)
    up_conv7 = concatenate([up_conv7, residual_path1], axis=3)
    up_conv8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer=ker_init, padding='same')(up_conv7)
    up_conv9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer=ker_init, padding='same')(up_conv8)
    shortcut7 = Conv2D(32, (1, 1), activation='relu', kernel_initializer=ker_init, padding='same')(up_conv7)
    residual_path7 = add([shortcut7, up_conv9])

    up_conv10 = UpSampling2D(size=(2, 2))(residual_path7)
    up_conv10 = concatenate([up_conv10, conv1], axis=3)
    up_conv11 = Conv2D(16, (3, 3), activation='relu', kernel_initializer=ker_init, padding='same')(up_conv10)
    up_conv12 = Conv2D(16, (3, 3), activation='relu', kernel_initializer=ker_init, padding='same')(up_conv11)
    shortcut8 = Conv2D(16, (1, 1), activation='relu', kernel_initializer=ker_init, padding='same')(up_conv10)
    residual_path8 = add([shortcut8, up_conv12])

    outputs = Conv2D(4, (1, 1), activation='softmax')(residual_path8)
    model = Model(inputs=[inputs], outputs=[outputs])
    model.summary()
    return model