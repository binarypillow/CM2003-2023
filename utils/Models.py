from tensorflow.keras.layers import (
    Dense,
    Flatten,
    MaxPooling2D,
    Conv2D,
    Conv2DTranspose,
    Activation,
    Dropout,
    SpatialDropout2D,
    Input,
    concatenate,
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
    extra_dense=False,
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
        Conv2D(
            filters=base * 8,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            name="Last_ConvLayer",
        )
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

    if extra_dense:
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


def conv_block(
    x,
    base,
    normalization=False,
):
    """Create a block with 2 convolutional layers.

    Args:
        x: previous block.
        base (int): the base number of neurons in conv2D layers.
        normalization (bool, optional): if True normalization layers are added after each conv2D layer. Defaults to False.

    Returns:
        object: the 2 layered block
    """

    x = Conv2D(
        filters=base,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        activation="relu",
    )(x)
    if normalization:
        x = BatchNormalization()(x)

    x = Conv2D(
        filters=base,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        activation="relu",
    )(x)
    if normalization:
        x = BatchNormalization()(x)
    return x


def encode(x, base, dropout=False, dropout_rate=0.2, normalization=False):
    """Create a contraction path block.

    Args:
        x: previous encode block.
        base (int): the base number of neurons in conv2D layers.
        dropout_rate (float, optional): the rate of the dropout layers. Defaults to 0.2.
        dropout (bool, optional): if True dropout layers are added after each dense layer. Defaults to False.

    Returns:
        object: the next encode block
    """

    l = conv_block(x, base, normalization)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(l)
    if dropout:
        x = Dropout(dropout_rate)(x)

    return x, l


def decode(
    x,
    x_encode,
    base,
    dropout=False,
    dropout_rate=0.2,
    normalization=False,
):
    """Create a expansion path block.

    Args:
        x: previous decode block.
        l_encode: related layer from encode path
        base (int): the base number of neurons in conv2D layers.
        dropout_rate (float, optional): the rate of the dropout layers. Defaults to 0.2.
        dropout (bool, optional): if True dropout layers are added after each dense layer. Defaults to False.

    Returns:
        object: the next decode block
    """

    x = Conv2DTranspose(
        filters=base, kernel_size=(2, 2), strides=(2, 2), padding="same"
    )(x)
    x = concatenate([x, x_encode])
    if dropout:
        x = Dropout(dropout_rate)(x)
    x = conv_block(x, base, normalization)

    return x


def UNet(
    img_ch,
    img_width,
    img_height,
    base,
    output_layers,
    depth=4,
    dropout=False,
    dropout_rate=0.2,
    normalization=False,
):
    """Create a U-Net model.

    Args:
        img_ch (int): the image channels.
        img_width (int): the image width.
        img_height (int): the image height.
        base (int): the base number of neurons in conv2D layers.
        output_layers (int, optional): the number of neurons in a output layer. Defaults to 1.
        dropout_rate (float, optional): the rate of the dropout layers. Defaults to 0.2.
        dropout (bool, optional): if True dropout layers are added after each dense layer. Defaults to False.

    Returns:
        object: the U-Net model
    """

    inputs = Input(shape=(img_width, img_height, img_ch))

    # Contraction path

    e1, l1 = encode(inputs, base, dropout, dropout_rate, normalization)
    e2, l2 = encode(e1, base * 2, dropout, dropout_rate, normalization)
    e3, l3 = encode(e2, base * 4, dropout, dropout_rate, normalization)
    e4, l4 = encode(e3, base * 8, dropout, dropout_rate, normalization)

    # Bottle neck

    x_bott = conv_block(e4, base * 16, normalization)

    # Expansion path

    d1 = decode(x_bott, l4, base * 8, dropout, dropout_rate, normalization)
    d2 = decode(d1, l3, base * 4, dropout, dropout_rate, normalization)
    d3 = decode(d2, l2, base * 2, dropout, dropout_rate, normalization)
    d4 = decode(d3, l1, base, dropout, dropout_rate, normalization)

    if output_layers == 1:
        outputs = Conv2D(
            filters=output_layers,
            kernel_size=(1, 1),
            padding="same",
            activation="sigmoid",
        )(d4)
    else:
        outputs = Conv2D(
            filters=output_layers,
            kernel_size=(1, 1),
            padding="same",
            activation="softmax",
        )(d4)

    model = Model(inputs, outputs, name="U-Net")

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
