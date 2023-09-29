from tensorflow.keras.layers import (
    Dense,
    Flatten,
    MaxPooling2D,
    Conv2D,
    Activation,
    Dropout,
    SpatialDropout2D,
    Input,
    BatchNormalization,
)
from tensorflow.keras import Sequential, Model


# LeNet model
def LeNet(
    img_ch,
    img_width,
    img_height,
    base,
    base_dense_1=64,
    dropout=False,
    dropout_rate=0.2,
    output_layers=1,
):
    """Create a LeNet model.

    Args:
        img_ch (int): the image channels.
        img_width (int): the image width.
        img_height (int): the image height.
        base (int): the base number of neurons in conv2D layers.
        base_dense_1 (int, optional): the base number of neurons in the dense layers. Defaults to 128.
        dropout (bool, optional): if True dropout layers are added after each dense layer. Defaults to False.
        dropout_rate (float, optional): the rate of the dropout layers. Defaults to 0.2.
        output_layers (int, optional): the number of neurons in a output layer. Defaults to 1.

    Returns:
        object: the LeNet model
    """
    model = Sequential()
    model.add(
        Conv2D(
            base,
            kernel_size=(3, 3),
            activation="relu",
            strides=1,
            padding="same",
            input_shape=(img_width, img_height, img_ch),
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(
        Conv2D(
            base * 2, kernel_size=(3, 3), activation="relu", strides=1, padding="same"
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(base_dense_1 * 2, activation="relu"))
    if dropout:
        model.add(Dropout(dropout_rate))
    if output_layers == 1:
        model.add(Dense(output_layers, activation="sigmoid"))
    else:
        model.add(Dense(output_layers, activation="softmax"))

    model.summary()
    return model


# AlexNet model
def AlexNet(
    img_ch,
    img_width,
    img_height,
    base,
    base_dense_1=128,
    base_dense_2=0,
    normalization=False,
    spatial_drop=False,
    spatial_drop_rate=0.1,
    dropout=False,
    dropout_rate=0.2,
    output_layers=1,
):
    """Create an AlexNet model.

    Args:
        img_ch (int): the image channels.
        img_width (int): the image width.
        img_height (int): the image height.
        base (int): the base number of neurons in conv2D layers.
        base_dense_1 (int, optional): the base number of neurons in the first dense layer. Defaults to 128.
        base_dense_2 (int, optional): the base number of neurons in the second dense layer. Defaults to 0.
        normalization (bool, optional): if True normalization layers are added after each conv2D layer. Defaults to False.
        spatial_drop (bool, optional): if True spatial dropout layers are added after each conv2D layer. Defaults to False.
        spatial_drop_rate (float, optional): the rate of the spatial dropout layers. Defaults to 0.1.
        dropout (bool, optional): if True dropout layers are added after each dense layer. Defaults to False.
        dropout_rate (float, optional): the rate of the dropout layers. Defaults to 0.2.
        output_layers (int, optional): the number of neurons in a output layer. Defaults to 1.

    Returns:
        object: the AlexNet model
    """
    model = Sequential()
    model.add(
        Conv2D(
            filters=base,
            input_shape=(img_width, img_height, img_ch),
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
        )
    )
    if normalization:
        model.add(BatchNormalization())
    if spatial_drop:
        model.add(SpatialDropout2D(spatial_drop_rate))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(
        Conv2D(filters=base * 2, kernel_size=(3, 3), strides=(1, 1), padding="same")
    )
    if normalization:
        model.add(BatchNormalization())
    if spatial_drop:
        model.add(SpatialDropout2D(spatial_drop_rate))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(
        Conv2D(filters=base * 4, kernel_size=(3, 3), strides=(1, 1), padding="same")
    )
    if normalization:
        model.add(BatchNormalization())
    if spatial_drop:
        model.add(SpatialDropout2D(spatial_drop_rate))
    model.add(Activation("relu"))
    model.add(
        Conv2D(filters=base * 4, kernel_size=(3, 3), strides=(1, 1), padding="same")
    )
    if normalization:
        model.add(BatchNormalization())
    if spatial_drop:
        model.add(SpatialDropout2D(spatial_drop_rate))
    model.add(Activation("relu"))
    model.add(
        Conv2D(filters=base * 2, kernel_size=(3, 3), strides=(1, 1), padding="same")
    )
    if normalization:
        model.add(BatchNormalization())
    if spatial_drop:
        model.add(SpatialDropout2D(spatial_drop_rate))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(base_dense_1))
    model.add(Activation("relu"))
    if dropout:
        model.add(Dropout(dropout_rate))
    if base_dense_2 != 0:
        model.add(Dense(base_dense_2))
        model.add(Activation("relu"))
        if dropout:
            model.add(Dropout(dropout_rate))
    model.add(Dense(output_layers))
    if output_layers == 1:
        model.add(Activation("sigmoid"))
    else:
        model.add(Activation("softmax"))
    model.summary()
    return model


# VGG16 model
def VGG16(
    img_ch,
    img_width,
    img_height,
    base,
    base_dense_1=64,
    dropout=False,
    dropout_rate=0.2,
    output_layers=1,
):
    """Create a VGG16 model.

    Args:
        img_ch (int): the image channels.
        img_width (int): the image width.
        img_height (int): the image height.
        base (int): the base number of neurons in conv2D layers.
        base_dense_1 (int, optional): the base number of neurons in the dense layers. Defaults to 128.
        dropout (bool, optional): if True dropout layers are added after each dense layer. Defaults to False.
        dropout_rate (float, optional): the rate of the dropout layers. Defaults to 0.2.
        output_layers (int, optional): the number of neurons in a output layer. Defaults to 1.

    Returns:
        object: the VGG16 model
    """
    model = Sequential()

    model.add(
        Conv2D(
            filters=base,
            input_shape=(img_width, img_height, img_ch),
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
        )
    )
    model.add(Activation("relu"))
    model.add(
        Conv2D(
            filters=base,
            input_shape=(img_width, img_height, img_ch),
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
        )
    )
    model.add(Activation("relu"))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(
        Conv2D(filters=base * 2, kernel_size=(3, 3), strides=(1, 1), padding="same")
    )
    model.add(Activation("relu"))
    model.add(
        Conv2D(filters=base * 2, kernel_size=(3, 3), strides=(1, 1), padding="same")
    )
    model.add(Activation("relu"))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(
        Conv2D(filters=base * 4, kernel_size=(3, 3), strides=(1, 1), padding="same")
    )
    model.add(Activation("relu"))
    model.add(
        Conv2D(filters=base * 4, kernel_size=(3, 3), strides=(1, 1), padding="same")
    )
    model.add(Activation("relu"))
    model.add(
        Conv2D(filters=base * 4, kernel_size=(3, 3), strides=(1, 1), padding="same")
    )
    model.add(Activation("relu"))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(
        Conv2D(filters=base * 8, kernel_size=(3, 3), strides=(1, 1), padding="same")
    )
    model.add(Activation("relu"))
    model.add(
        Conv2D(filters=base * 8, kernel_size=(3, 3), strides=(1, 1), padding="same")
    )
    model.add(Activation("relu"))
    model.add(
        Conv2D(filters=base * 8, kernel_size=(3, 3), strides=(1, 1), padding="same")
    )
    model.add(Activation("relu"))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(
        Conv2D(filters=base * 8, kernel_size=(3, 3), strides=(1, 1), padding="same")
    )
    model.add(Activation("relu"))
    model.add(
        Conv2D(filters=base * 8, kernel_size=(3, 3), strides=(1, 1), padding="same")
    )
    model.add(Activation("relu"))
    model.add(
        Conv2D(filters=base * 8, kernel_size=(3, 3), strides=(1, 1), padding="same")
    )
    model.add(Activation("relu"))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(base_dense_1))
    model.add(Activation("relu"))
    if dropout:
        model.add(Dropout(dropout_rate))

    model.add(Dense(base_dense_1))
    model.add(Activation("relu"))
    if dropout:
        model.add(Dropout(dropout_rate))

    model.add(Dense(output_layers))
    if output_layers == 1:
        model.add(Activation("sigmoid"))
    else:
        model.add(Activation("softmax"))
    model.summary()

    return model


# costum MLP model
def MyModel(img_width, img_height, img_ch, base_dense):
    """Create a custom model using the functional API.

    Args:
        img_ch (int): the image channels.
        img_width (int): the image width.
        img_height (int): the image height.
        base_dense (int): the base number of neurons in dense layers.

    Returns:
        object: the custom model
    """
    input_size = (img_width, img_height, img_ch)
    inputs_layer = Input(shape=input_size, name="input_layer")  # input layer
    flatten = Flatten()

    dense1 = Dense(base_dense, activation="relu")  # Base layer
    dense2 = Dense(base_dense // 2, activation="relu")  # First hidden layer
    dense3 = Dense(base_dense // 4, activation="relu")  # Second hidden layer
    dense4 = Dense(1, activation="sigmoid")  # Third hidden layer

    x = flatten(inputs_layer)
    x = dense1(x)
    x = dense2(x)
    x = dense3(x)
    out = dense4(x)

    model = Model(inputs=inputs_layer, outputs=out)
    model.summary()
    return model
