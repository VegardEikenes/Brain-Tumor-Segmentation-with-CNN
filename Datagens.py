import tensorflow as tf
from tensorflow import keras
import nibabel as nib
import numpy as np
import os
import cv2

PATH = 'C:/Users/VegardEikenes/Desktop/Bachelor/Data/BraTS2021_TrainingData/'
PATH3D = 'C:/Users/VegardEikenes/Desktop/Bachelor/Data/3Ddata/'
VOLUME_SLICES = 100
VOLUME_START_AT = 22


"""
This generator was originally written by Rastislav: https://www.kaggle.com/code/rastislav/3d-mri-brain-tumor-segmentation-u-net 
with inspiration from: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

The generator has been modified to include an increased number of modalities, and can easily be
adapted. 
Changes made include: 
* __data_generation function is changed by adding an increased number of modalities for experimentation.
"""


class DataGenerator2D(keras.utils.Sequence):
    def __init__(self, list_IDs, dim=(128, 128), batch_size=1, n_channels=3, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        Batch_ids = [self.list_IDs[k] for k in indexes]

        X, y = self.__data_generation(Batch_ids)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, Batch_ids):
        X = np.zeros((self.batch_size * VOLUME_SLICES, *self.dim, self.n_channels))
        y = np.zeros((self.batch_size * VOLUME_SLICES, 240, 240))
        Y = np.zeros((self.batch_size * VOLUME_SLICES, *self.dim, 4))

        for c, i in enumerate(Batch_ids):
            case_path = os.path.join(PATH, i)

            data_path = os.path.join(case_path, f'{i}_flair.nii')
            flair = nib.load(data_path).get_fdata()

            data_path = os.path.join(case_path, f'{i}_t1ce.nii')
            t1ce = nib.load(data_path).get_fdata()

            data_path = os.path.join(case_path, f'{i}_t2.nii')
            t2 = nib.load(data_path).get_fdata()

            #             data_path = os.path.join(case_path, f'{i}_t1.nii')
            #             t1 = nib.load(data_path).get_fdata()

            data_path = os.path.join(case_path, f'{i}_seg.nii')
            seg = nib.load(data_path).get_fdata()

            for j in range(VOLUME_SLICES):
                X[j + VOLUME_SLICES * c, :, :, 0] = cv2.resize(flair[:, :, j + VOLUME_START_AT], self.dim)
                X[j + VOLUME_SLICES * c, :, :, 1] = cv2.resize(t1ce[:, :, j + VOLUME_START_AT], self.dim)
                X[j + VOLUME_SLICES * c, :, :, 2] = cv2.resize(t2[:, :, j + VOLUME_START_AT], self.dim)
                #                 X[j +VOLUME_SLICES*c,:,:,2] = cv2.resize(t1[:,:,j+VOLUME_START_AT], dim)

                y[j + VOLUME_SLICES * c] = seg[:, :, j + VOLUME_START_AT]

        y[y == 4] = 3
        mask = tf.one_hot(y, 4)
        Y = tf.image.resize(mask, self.dim)
        return X / np.max(X), Y


"""
This 3D generator is inspired from the same sources as the datagenerator above. 
https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
https://www.kaggle.com/code/rastislav/3d-mri-brain-tumor-segmentation-u-net

The generator is adapted to generate 3D (128x128x128) volumes instead of 2D volumes on the fly while training. The
number of modalities to include can easily be adapted. The generator does however only accept batch
sizes of 1. This is not an issue for my project as increasing the batch size to 2 would 
cause memory exhaustion. 
"""


class DataGenerator3D(keras.utils.Sequence):
    def __init__(self, list_IDs, batch_size=1, n_channels=3, shuffle=True):
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        Batch_ids = [self.list_IDs[k] for k in indexes]

        X, y = self.__data_generation(Batch_ids)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, Batch_ids):
        for c, i in enumerate(Batch_ids):
            case_path = os.path.join(PATH, i)

            data_path = os.path.join(case_path, f'{i}_flair.nii')
            flair = nib.load(data_path).get_fdata()

            data_path = os.path.join(case_path, f'{i}_t1ce.nii')
            t1ce = nib.load(data_path).get_fdata()

            data_path = os.path.join(case_path, f'{i}_t2.nii')
            t2 = nib.load(data_path).get_fdata()

            #             data_path = os.path.join(case_path, f'{i}_t1.nii')
            #             t1 = nib.load(data_path).get_fdata()

            data_path = os.path.join(case_path, f'{i}_seg.nii')
            seg = nib.load(data_path).get_fdata()
            seg[seg == 4] = 3

            stack = np.stack([flair, t1ce, t2], axis=3)
            stack = stack[56:184, 56:184, 13:141]
            seg = seg[56:184, 56:184, 13:141]

        Y = tf.one_hot(seg, 4)
        Y = tf.expand_dims(Y, 0)
        X = np.expand_dims(stack, 0)
        return X / np.max(X), Y