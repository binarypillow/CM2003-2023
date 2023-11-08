from keras import backend as K


"""
------------------------
    DICE COEFFICIENT
------------------------
"""


# dice coefficent
def dice_coeff(y_true, y_pred, smooth=0.001):
    """Calculate dice coefficient

    Args:
        y_true (nd-array): ground truth mask.
        y_pred (nd-array): predicted mask.
        smooth (float, optional): smoothing term. Defaults to 0.001.

    Returns:
        tensor: dice coeff value.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice


# dice coefficent with multiple prediction
def dice_coeff_multilabel(y_true, y_pred):
    """Calculate multilabel dice coefficient

    Args:
        y_true (nd-array): ground truth mask.
        y_pred (nd-array): predicted mask.

    Returns:
        tensor: dice coeff multilabel value.
    """
    numClass = y_pred.shape[3]
    dice = 0
    for index in range(1, numClass):
        dice += dice_coeff(y_true[:, :, :, index], y_pred[:, :, :, index])
    dice = dice / (numClass - 1)
    return dice


# dice coefficent with arteries prediction
def dice_coeff_arteries(y_true, y_pred):
    """Calculate dice coefficient relative to the arteries

    Args:
        y_true (nd-array): ground truth mask.
        y_pred (nd-array): predicted mask.

    Returns:
        tensor: dice coeff arteries value.
    """
    dice = dice_coeff(y_true[:, :, :, 1], y_pred[:, :, :, 1])
    return dice


# dice coefficent with veins prediction
def dice_coeff_veins(y_true, y_pred):
    """Calculate dice coefficient relative to the veins

    Args:
        y_true (nd-array): ground truth mask.
        y_pred (nd-array): predicted mask.

    Returns:
        tensor: dice coeff veins value.
    """
    dice = dice_coeff(y_true[:, :, :, 2], y_pred[:, :, :, 2])
    return dice


# dice loss
def dice_loss(y_true, y_pred):
    """Calculate dice loss

    Args:
        y_true (nd-array): ground truth mask.
        y_pred (nd-array): predicted mask.

    Returns:
        tensor: dice loss.
    """
    numClass = y_pred.shape[3]
    if numClass == 1:
        loss = -dice_coeff(y_true, y_pred)
    else:
        loss = -dice_coeff_multilabel(y_true, y_pred)
    return loss


"""
---------------------------
    JACCARD COEFFICIENT
---------------------------
"""


# jaccard coefficent
def jaccard_coeff(y_true, y_pred):
    """Calculate dice coefficient

    Args:
        y_true (nd-array): ground truth mask.
        y_pred (nd-array): predicted mask.
        smooth (float, optional): smoothing term. Defaults to 0.001.

    Returns:
        tensor: dice coeff value.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return intersection / union


# jaccard coefficent with multiple prediction
def jaccard_coeff_multilabel(y_true, y_pred):
    """Calculate multilabel jaccard coefficient

    Args:
        y_true (nd-array): ground truth mask.
        y_pred (nd-array): predicted mask.

    Returns:
        tensor: dice coeff multilabel value.
    """
    numClass = y_pred.shape[3]
    jaccard = 0
    for index in range(1, numClass):
        jaccard += jaccard_coeff(y_true[:, :, :, index], y_pred[:, :, :, index])
    jaccard = jaccard / (numClass - 1)
    return jaccard


# jaccard coefficent with arteries prediction
def jaccard_coeff_arteries(y_true, y_pred):
    """Calculate jaccard coefficient relative to arteries

    Args:
        y_true (nd-array): ground truth mask.
        y_pred (nd-array): predicted mask.

    Returns:
        tensor: jaccard veins coefficient value.
    """
    jaccard = jaccard_coeff(y_true[:, :, :, 1], y_pred[:, :, :, 1])
    return jaccard


# jaccard coefficent with veins prediction
def jaccard_coeff_veins(y_true, y_pred):
    """Calculate jaccard coefficient relative to veins

    Args:
        y_true (nd-array): ground truth mask.
        y_pred (nd-array): predicted mask.

    Returns:
        tensor: jaccard veins coefficient value.
    """
    jaccard = jaccard_coeff(y_true[:, :, :, 2], y_pred[:, :, :, 2])
    return jaccard
