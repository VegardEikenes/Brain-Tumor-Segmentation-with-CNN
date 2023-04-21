from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import tensorflow as tf
from tensorflow_addons.layers import *
tf.keras.backend.set_image_data_format('channels_last')

"""
This code is inspired by Naomi Fridman: https://naomi-fridman.medium.com/multi-class-image-segmentation-a5cc671e647a and
has been adapted to my project. 

Changes made to the code:
* Number of filters used in each layer is decreased to reduce the size of the model
* The structure is modified
* Activation functions are experimented with in every convolutional layer, e.g., LeakyRelu or PRelu implemented instead of Relu
* Dropout layers are experimented with for each model trained by adding/removing dropout in every block
* Batch normalization and instance normalization is experimented with for each model trained by adding/removing it in every block

An example of how PReLU was incorporated is demnstrated below and commented out
"""
def build_unet(n_channels, ker_init, dropout):
    inputs = Input((128, 128, n_channels))

    conv1 = Conv2D(16, 3, padding='same', kernel_initializer=ker_init)(inputs)
    conv1 = InstanceNormalization(axis=-1)(conv1)
    conv1 = PReLU(alpha_initializer=tf.initializers.constant(0.25))(conv1)
    # conv1 = Dropout(dropout)(conv1)
    conv1 = Conv2D(16, 3, padding='same', kernel_initializer=ker_init)(conv1)
    conv1 = InstanceNormalization(axis=-1)(conv1)
    conv1 = PReLU(alpha_initializer=tf.initializers.constant(0.25))(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(32, 3, padding='same', kernel_initializer=ker_init)(pool1)
    conv2 = InstanceNormalization(axis=-1)(conv2)
    conv2 = PReLU(alpha_initializer=tf.initializers.constant(0.25))(conv2)
    # conv2 = Dropout(dropout)(conv2)
    conv2 = Conv2D(32, 3, padding='same', kernel_initializer=ker_init)(conv2)
    conv2 = InstanceNormalization(axis=-1)(conv2)
    conv2 = PReLU(alpha_initializer=tf.initializers.constant(0.25))(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(64, 3, padding='same', kernel_initializer=ker_init)(pool2)
    conv3 = InstanceNormalization(axis=-1)(conv3)
    conv3 = PReLU(alpha_initializer=tf.initializers.constant(0.25))(conv3)
    # conv3 = Dropout(dropout)(conv3)
    conv3 = Conv2D(64, 3, padding='same', kernel_initializer=ker_init)(conv3)
    conv3 = InstanceNormalization(axis=-1)(conv3)
    conv3 = PReLU(alpha_initializer=tf.initializers.constant(0.25))(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(128, 3, padding='same', kernel_initializer=ker_init)(pool3)
    conv4 = InstanceNormalization(axis=-1)(conv4)
    conv4 = PReLU(alpha_initializer=tf.initializers.constant(0.25))(conv4)
    # conv4 = Dropout(dropout)(conv4)
    conv4 = Conv2D(128, 3, padding='same', kernel_initializer=ker_init)(conv4)
    conv4 = InstanceNormalization(axis=-1)(conv4)
    conv4 = PReLU(alpha_initializer=tf.initializers.constant(0.25))(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(256, 3, padding='same', kernel_initializer=ker_init)(pool4)
    conv5 = InstanceNormalization(axis=-1)(conv5)
    conv5 = PReLU(alpha_initializer=tf.initializers.constant(0.25))(conv5)
    # conv5 = Dropout(dropout)(conv5)
    conv5 = Conv2D(256, 3, padding='same', kernel_initializer=ker_init)(conv5)
    conv5 = InstanceNormalization(axis=-1)(conv5)
    conv5 = PReLU(alpha_initializer=tf.initializers.constant(0.25))(conv5)


    up6 = Conv2D(128, 2, padding='same', kernel_initializer=ker_init) \
        (UpSampling2D(size=(2, 2))(conv5))
    merge6 = concatenate([conv4, up6], axis=3)
    conv6 = Conv2D(128, 3, padding='same', kernel_initializer=ker_init)(merge6)
    conv6 = InstanceNormalization(axis=-1)(conv6)
    # conv6 = Dropout(dropout)(conv6)
    conv6 = Conv2D(128, 3, padding='same', kernel_initializer=ker_init)(conv6)
    conv6 = InstanceNormalization(axis=-1)(conv6)
    conv6 = PReLU(alpha_initializer=tf.initializers.constant(0.25))(conv6)

    up7 = Conv2D(64, 2, padding='same', kernel_initializer=ker_init) \
        (UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(64, 3, padding='same', kernel_initializer=ker_init)(merge7)
    conv7 = InstanceNormalization(axis=-1)(conv7)
    conv7 = PReLU(alpha_initializer=tf.initializers.constant(0.25))(conv7)
    # conv7 = Dropout(dropout)(conv7)
    conv7 = Conv2D(64, 3, padding='same', kernel_initializer=ker_init)(conv7)
    conv7 = InstanceNormalization(axis=-1)(conv7)
    conv7 = PReLU(alpha_initializer=tf.initializers.constant(0.25))(conv7)

    up8 = Conv2D(32, 2, padding='same', kernel_initializer=ker_init) \
        (UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(32, 3, padding='same', kernel_initializer=ker_init)(merge8)
    conv8 = InstanceNormalization(axis=-1)(conv8)
    conv8 = PReLU(alpha_initializer=tf.initializers.constant(0.25))(conv8)
    # conv8 = Dropout(dropout)(conv8)
    conv8 = Conv2D(32, 3, padding='same', kernel_initializer=ker_init)(conv8)
    conv8 = InstanceNormalization(axis=-1)(conv8)
    conv8 = PReLU(alpha_initializer=tf.initializers.constant(0.25))(conv8)

    up9 = Conv2D(16, 2, padding='same', kernel_initializer = ker_init)(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(16, 3, padding='same', kernel_initializer=ker_init)(merge9)
    conv9 = InstanceNormalization(axis=-1)(conv9)
    conv9 = PReLU(alpha_initializer=tf.initializers.constant(0.25))(conv9)
    # conv9 = Dropout(dropout)(conv9)
    conv9 = Conv2D(16, 3, padding='same', kernel_initializer=ker_init)(conv9)
    conv9 = InstanceNormalization(axis=-1)(conv9)
    conv9 = PReLU(alpha_initializer=tf.initializers.constant(0.25))(conv9)

    outputs = Conv2D(4, (1, 1), activation='softmax')(conv9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    model.summary()
    return model

