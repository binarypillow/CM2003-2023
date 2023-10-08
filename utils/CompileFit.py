from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K
from tensorflow import reshape, reduce_sum

import absl.logging

absl.logging.set_verbosity(absl.logging.ERROR)

"""
def dice_coeff(y_true, y_pred):
    smooth = 1.0
    # Flatten
    y_true_f = reshape(y_true, [-1])
    y_pred_f = reshape(y_pred, [-1])
    intersection = reduce_sum(y_true_f * y_pred_f)
    score = (2.0 * intersection + smooth) / (
        reduce_sum(y_true_f) + reduce_sum(y_pred_f) + smooth
    )
    return score

"""


def dice_coeff(y_true, y_pred, smooth=0.1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice


def dice_loss(y_true, y_pred):
    loss = -dice_coeff(y_true, y_pred)
    return loss


# compile and fit the model
def compileFit(
    model,
    loss,
    opt,
    metrics,
    learning_rate,
    batch_size,
    epochs,
    task_name,
    iter_name="",
    x_train=None,
    y_train=None,
    x_test=None,
    y_test=None,
    train_generator=None,
    valid_generator=None,
    steps_per_epoch=None,
    validation_steps=None,
):
    """Compile and fit an untrained model.

    Args:
        model (object): the untrained model.
        loss (string): the name of the loss function.
        opt (string): the name of the chosen optimizer.
        metrics (string): the name of the metrics to calculate.
        x_train (Numpy array): train data.
        y_train (Numpy array): labels of train data.
        x_test (Numpy array): test data.
        y_test (Numpy array): labels of test data.
        learning_rate (float): the learning rate.
        batch_size (int): the batch size.
        epochs (_type_): the number of epochs.
        task_name (str): the folder in which the checkpoint is saved.
        iter_name (str, optional): an optional extention to the default file name. Defaults to "".
    Returns:
        _type_: _description_
    """
    if opt == "Adam":
        optimizer = Adam(learning_rate=learning_rate)
    elif opt == "SGD":
        optimizer = SGD(learning_rate=learning_rate)
    else:
        optimizer = RMSprop(learning_rate=learning_rate)

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    # Saving the weights of the model
    checkpoint_path = f"results/{task_name}/cp{iter_name}.ckpt"
    cp_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        save_best_only=True,
        save_weights_only=True,
        verbose=2,
    )
    if not train_generator:
        model_fit = model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=[x_test, y_test],
            verbose=1,
            callbacks=[cp_callback],
        )
    else:
        model_fit = model.fit(
            train_generator,
            batch_size=batch_size,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            validation_data=valid_generator,
            verbose=1,
            callbacks=[cp_callback],
        )

    return model_fit
