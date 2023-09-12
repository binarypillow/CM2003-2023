from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Conv2D, Activation, Dropout, Input
from tensorflow.keras import Sequential, Model

def MyModel(img_width, img_height, img_ch, base_dense):
  """
  Functional API model.
  name the last layer as "out"; e.g., out = ....
  """
  input_size = (img_width, img_height, img_ch)
  inputs_layer = Input(shape=input_size, name='input_layer')  # input layer
  flatten = Flatten()
  
  dense1 = Dense(base_dense, activation='relu')      # Base layer
  dense2 = Dense(base_dense//2, activation='relu')   # First hidden layer
  dense3 = Dense(base_dense//4, activation='relu')   # Second hidden layer
  dense4 = Dense(1, activation='sigmoid')            # Third hidden layer
  
  x = flatten(inputs_layer)
  x = dense1(x) 
  x = dense2(x) 
  x = dense3(x) 
  out = dense4(x)

  model = Model(inputs=inputs_layer, outputs=out)
  model.summary()
  return model

# LeNet model
def LeNet(img_ch, img_width, img_height, base):
    '''
    Parameters
    ----------
    img_ch : Int
    The image channels.
    img_width : Int
    The image width.
    img_height : Int
    The image height.
    base : Int
    The neuron base number

    Returns
    -------
    model : object 
    LeNet Network model.
    '''
    model = Sequential()
    model.add(Conv2D(base, kernel_size = (3, 3), activation='relu',strides=1, padding='same', input_shape = (img_width, img_height, img_ch)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(base*2, kernel_size = (3, 3), activation='relu', strides=1, padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(base*2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    return model

# AlexNet model
def AlexNet(img_ch, img_width, img_height, base):
    '''
    Parameters
    ----------
    img_ch : Int
    The image channels.
    img_width : Int
    The image width.
    img_height : Int
    The image height.
    base : Int
    The neuron base number

    Returns
    -------
    model : object 
    AlexNet Network model.
    '''
    model = Sequential()
    model.add(Conv2D(filters=base, input_shape=(img_width, img_height, img_ch), kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=base *2, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=base *4, kernel_size=(3,3), strides=(1,1), padding='same')) 
    model.add(Activation('relu'))
    model.add(Conv2D(filters=base *4, kernel_size=(3,3), strides=(1,1), padding='same')) 
    model.add(Activation('relu'))
    model.add(Conv2D(filters=base *2, kernel_size=(3,3), strides=(1,1), padding='same')) 
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.summary()
    return model

# AlexNet modelwith drop out layer
def AlexNet1(img_ch, img_width, img_height, base, dropout_rate):
    '''
    Parameters
    ----------
    img_ch : Int
    The image channels.
    img_width : Int
    The image width.
    img_height : Int
    The image height.
    base : Int
    The neuron base number

    Returns
    -------
    model : object 
    AlexNet Network model with a “drop out layer” after each dense layer.
    '''
    model = Sequential()
    model.add(Conv2D(filters=base, input_shape=(img_width, img_height, img_ch), kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=base *2, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=base *4, kernel_size=(3,3), strides=(1,1), padding='same')) 
    model.add(Activation('relu'))
    model.add(Conv2D(filters=base *4, kernel_size=(3,3), strides=(1,1), padding='same')) 
    model.add(Activation('relu'))
    model.add(Conv2D(filters=base *2, kernel_size=(3,3), strides=(1,1), padding='same')) 
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.summary()
    return model


