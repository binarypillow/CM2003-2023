from keras.layers import (
    MaxPooling2D,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    Input,
    concatenate,
    BatchNormalization,
)
from keras import Model
from keras.applications import VGG16

"""
-------------
    U-NET
-------------
"""


# convolutional layers block
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
        object: the 2 layered block.
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


# encoding
def encode(x, base, dropout=False, dropout_rate=0.2, normalization=False):
    """Create a contraction path block.

    Args:
        x: previous encode block.
        base (int): the base number of neurons in conv2D layers.
        dropout_rate (float, optional): the rate of the dropout layers. Defaults to 0.2.
        dropout (bool, optional): if True dropout layers are added after each dense layer. Defaults to False.

    Returns:
        object: the next encode block.
    """

    l = conv_block(x, base, normalization)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(l)
    if dropout:
        x = Dropout(dropout_rate)(x)

    return x, l


# decoding
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
        x_encode: related layer from encode path.
        base (int): the base number of neurons in conv2D layers.
        dropout_rate (float, optional): the rate of the dropout layers. Defaults to 0.2.
        dropout (bool, optional): if True dropout layers are added after each dense layer. Defaults to False.

    Returns:
        object: the next decode block.
    """

    x = Conv2DTranspose(
        filters=base, kernel_size=(2, 2), strides=(2, 2), padding="same"
    )(x)
    x = concatenate([x, x_encode])
    if dropout:
        x = Dropout(dropout_rate)(x)
    x = conv_block(x, base, normalization)

    return x


# UNet
def UNet(
    img_ch,
    img_width,
    img_height,
    base,
    output_layers,
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
        normalization (bool, optional): if True normalization layers are added after each dense layer. Default to False.

    Returns:
        object: the U-Net model.
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

    model = Model(inputs, outputs=outputs)
    return model


# UNet + transfer learning (VGG16)
def UNetVGG16(
    img_ch,
    img_width,
    img_height,
    base,
    output_layers,
    dropout=False,
    dropout_rate=0.2,
    normalization=False,
):
    """Create a U-Net model with VGG16 layers as encoders.

    Args:
        img_ch (int): the image channels.
        img_width (int): the image width.
        img_height (int): the image height.
        base (int): the base number of neurons in conv2D layers.
        output_layers (int, optional): the number of neurons in a output layer. Defaults to 1.
        dropout_rate (float, optional): the rate of the dropout layers. Defaults to 0.2.
        dropout (bool, optional): if True dropout layers are added after each dense layer. Defaults to False.
        normalization (bool, optional): if True normalization layers are added after each dense layer. Default to False.

    Returns:
        object: the U-Net model.
    """
    inputs = Input(shape=(img_width, img_height, img_ch))

    # VGG16 model without top layers
    vgg16 = VGG16(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
    )

    # Output tensor from the VGG16 backbone
    e1 = vgg16.get_layer("block1_conv2").output
    e2 = vgg16.get_layer("block2_conv2").output
    e3 = vgg16.get_layer("block3_conv3").output
    e4 = vgg16.get_layer("block4_conv3").output

    # Bottle neck
    x_bott = vgg16.get_layer("block5_conv3").output

    # Expansion path
    d1 = decode(x_bott, e4, base * 8, dropout, dropout_rate, normalization)
    d2 = decode(d1, e3, base * 4, dropout, dropout_rate, normalization)
    d3 = decode(d2, e2, base * 2, dropout, dropout_rate, normalization)
    d4 = decode(d3, e1, base, dropout, dropout_rate, normalization)

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

    model = Model(inputs=inputs, outputs=outputs)

    return model
