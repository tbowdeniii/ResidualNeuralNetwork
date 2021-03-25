#https://drive.google.com/file/d/1PcrwYlWFZeTJqcHpUqwsPMZGZWqh4Kb-/view?usp=sharing

#Author: Thomas Bowden
#Date: 03-13-21
#Course: CSC 4700


import tensorflow as tf
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, BatchNormalization, ReLU, Flatten, Dense, Add
from tensorflow.keras import Input
from tensorflow.keras.models import load_model

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    print(e)


cifar10 = tf.keras.datasets.cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print(type(x_train), x_train.shape)
print(np.min(x_train), np.max(x_train))





def model(input_shape = (32, 32, 3)):

    x_inputs = Input(shape = input_shape)
    x = Conv2D(32, 3, strides = 1, padding = 'same')(x_inputs)
    x = BatchNormalization(axis = -1)(x)
    x = tf.keras.layers.Activation('relu')(x)




    # Section A - First Residual Block
    first_rb_input = x
    y = first_rb_input

    x = Conv2D(32, 3, strides = 1, padding = 'same')(first_rb_input)
    x = BatchNormalization(axis = -1)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = Conv2D(32, 3, strides = 1, padding = 'same')(x)
    x = BatchNormalization(axis = -1)(x)

    x = Add()([x, y])
    x = tf.keras.layers.Activation('relu')(x)


    # Section A - Second Residual Block
    first_rb_input = x
    y = first_rb_input

    x = Conv2D(32, 3, strides = 1, padding = 'same')(first_rb_input)
    x = BatchNormalization(axis = -1)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = Conv2D(32, 3, strides = 1, padding = 'same')(x)
    x = BatchNormalization(axis = -1)(x)

    x = Add()([x, y])
    x = tf.keras.layers.Activation('relu')(x)


    # Section A - Third Residual Block
    first_rb_input = x
    y = first_rb_input

    x = Conv2D(32, 3, strides = 1, padding = 'same')(first_rb_input)
    x = BatchNormalization(axis = -1)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = Conv2D(32, 3, strides = 1, padding = 'same')(x)
    x = BatchNormalization(axis = -1)(x)

    x = Add()([x, y])
    x = tf.keras.layers.Activation('relu')(x)




    # Section B - First Residual Block
    first_rb_input = x
    y = first_rb_input

    x = Conv2D(64, 3, strides = 2, padding = 'same')(first_rb_input)
    x = BatchNormalization(axis = -1)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = Conv2D(64, 3, strides = 1, padding = 'same')(x)
    x = BatchNormalization(axis = -1)(x)

    # Change Shape of skip-tensor to match result tensor
    y = Conv2D(64, 1, strides = 2, padding = 'same')(first_rb_input)

    x = Add()([x, y])
    x = tf.keras.layers.Activation('relu')(x)


    # Section B - Second Residual Block
    first_rb_input = x
    y = first_rb_input

    x = Conv2D(64, 3, strides = 1, padding = 'same')(first_rb_input)
    x = BatchNormalization(axis = -1)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = Conv2D(64, 3, strides = 1, padding = 'same')(x)
    x = BatchNormalization(axis = -1)(x)

    x = Add()([x, y])
    x = tf.keras.layers.Activation('relu')(x)


    # Section B - Third Residual Block
    first_rb_input = x
    y = first_rb_input

    x = Conv2D(64, 3, strides = 1, padding = 'same')(first_rb_input)
    x = BatchNormalization(axis = -1)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = Conv2D(64, 3, strides = 1, padding = 'same')(x)
    x = BatchNormalization(axis = -1)(x)

    x = Add()([x, y])
    x = tf.keras.layers.Activation('relu')(x)




    # Section C - First Residual Block
    first_rb_input = x
    y = first_rb_input

    x = Conv2D(128, 3, strides = 2, padding = 'same')(first_rb_input)
    x = BatchNormalization(axis = -1)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = Conv2D(128, 3, strides = 1, padding = 'same')(x)
    x = BatchNormalization(axis = -1)(x)

    # Change Shape of skip-tensor to match result tensor
    y = Conv2D(128, 1, strides = 2, padding = 'same')(first_rb_input)

    x = Add()([x, y])
    x = tf.keras.layers.Activation('relu')(x)


    # Section C - Second Residual Block
    first_rb_input = x
    y = first_rb_input

    x = Conv2D(128, 3, strides = 1, padding = 'same')(first_rb_input)
    x = BatchNormalization(axis = -1)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = Conv2D(128, 3, strides = 1, padding = 'same')(x)
    x = BatchNormalization(axis = -1)(x)

    x = Add()([x, y])
    x = tf.keras.layers.Activation('relu')(x)


    # Section C - Third Residual Block
    first_rb_input = x
    y = first_rb_input

    x = Conv2D(128, 3, strides = 1, padding = 'same')(first_rb_input)
    x = BatchNormalization(axis = -1)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = Conv2D(128, 3, strides = 1, padding = 'same')(x)
    x = BatchNormalization(axis = -1)(x)

    x = Add()([x, y])
    x = tf.keras.layers.Activation('relu')(x)



    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(10, activation = 'softmax')(x)

    model = tf.keras.models.Model(inputs= x_inputs, outputs= x, name = 'ResNet')

    return model

ResNet = model(input_shape = (32, 32, 3))

ResNet.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

history = ResNet.fit(x_train, y_train, batch_size = 128, epochs = 5, validation_data = (x_test, y_test))
tloss, tacc = ResNet.evaluate(x_test, y_test, verbose = 2)
print('Acccuracy on test data is:', tacc)




def create_model():
    return ResNet

def get_trained_model():
    ResNet.save('ResNet.h5')
    trained_model = load_model('ResNet.h5')
    return trained_model
