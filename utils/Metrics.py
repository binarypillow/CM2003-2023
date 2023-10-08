from tensorflow import reshape, reduce_sum


def dice_coeff(y_true, y_pred):
    """Calculate score coefficient.

    Args:
        y_true: the true values.
        y_pred: the predicted values.
    Returns:
        float: the dice coefficient value.
    """
    smooth = 1.0
    # Flatten
    y_true_f = reshape(y_true, [-1])
    y_pred_f = reshape(y_pred, [-1])
    intersection = reduce_sum(y_true_f * y_pred_f)
    score = (2.0 * intersection + smooth) / (
        reduce_sum(y_true_f) + reduce_sum(y_pred_f) + smooth
    )
    return score


def dice_loss(y_true, y_pred):
    """Calculate loss.

    Args:
        y_true: true values.
        y_pred: predicted values.

    Returns:
        float: the dice loss value.
    """
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss
