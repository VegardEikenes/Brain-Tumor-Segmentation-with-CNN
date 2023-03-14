import keras.backend as K

"""
This code is originally written by Rastislav: https://www.kaggle.com/code/rastislav/3d-mri-brain-tumor-segmentation-u-net
with inspiration from https://github.com/keras-team/keras/issues/9395. 
These metrics have been modified to work on 3D volumes.
Changes made to the code:
* Added dimension to every function.
"""


def dice_coef(y_true, y_pred, smooth=1.0):
    class_num = 4
    for i in range(class_num):
        y_true_f = K.flatten(y_true[:, :, :, :, i])
        y_pred_f = K.flatten(y_pred[:, :, :, :, i])
        intersection = K.sum(y_true_f * y_pred_f)
        loss = ((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
        if i == 0:
            total_loss = loss
        else:
            total_loss = total_loss + loss
    total_loss = total_loss / class_num
    return total_loss


def dice_coef_necrotic(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:, :, :, :, 1] * y_pred[:, :, :, :, 1]))
    return (2. * intersection) / (
                K.sum(K.square(y_true[:, :, :, :, 1])) + K.sum(K.square(y_pred[:, :, :, :, 1])) + epsilon)


def dice_coef_edema(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:, :, :, :, 2] * y_pred[:, :, :, :, 2]))
    return (2. * intersection) / (
                K.sum(K.square(y_true[:, :, :, :, 2])) + K.sum(K.square(y_pred[:, :, :, :, 2])) + epsilon)


def dice_coef_enhancing(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:, :, :, :, 3] * y_pred[:, :, :, :, 3]))
    return (2. * intersection) / (
                K.sum(K.square(y_true[:, :, :, :, 3])) + K.sum(K.square(y_pred[:, :, :, :, 3])) + epsilon)