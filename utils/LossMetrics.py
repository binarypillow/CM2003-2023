import numpy as np
from tensorflow.keras import backend as K


def dice_coeff(y_true, y_pred, smooth=0.1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice


def dice_loss(y_true, y_pred):
    loss = -dice_coeff(y_true, y_pred)
    return loss


def weighted_loss(weight_map, weight_strength):
    def weighted_dice_loss(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        weight_f = K.flatten(weight_map)
        y_pred_f = K.flatten(y_pred)
        weight_f = weight_f * weight_strength + 1
        wy_true_f = y_true_f * weight_f
        wy_pred_f = y_pred_f * weight_f
        return -dice_coeff(wy_true_f, wy_pred_f)

    return weighted_dice_loss
