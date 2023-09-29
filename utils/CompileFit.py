from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint
import absl.logging

absl.logging.set_verbosity(absl.logging.ERROR)


# compile and fit the model
def compileFit(
    model,
    loss,
    opt,
    metrics,
    x_train,
    y_train,
    x_test,
    y_test,
    learning_rate,
    batch_size,
    epochs,
    task_name,
    iter_name="",
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

    model_fit = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=[x_test, y_test],
        verbose=1,
        callbacks=[cp_callback],
    )

    return model_fit
