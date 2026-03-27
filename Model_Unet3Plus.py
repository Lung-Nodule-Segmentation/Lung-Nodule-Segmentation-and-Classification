import cv2
from tensorflow import keras
from keras import layers
import numpy as np


def unet_3plus(input_shape, num_classes):
    def conv_block(x, filters, kernel_size=(3, 3), activation='relu', padding='same'):
        x = layers.Conv2D(filters, kernel_size, activation=activation, padding=padding)(x)
        x = layers.Conv2D(filters, kernel_size, activation=activation, padding=padding)(x)
        return x

    # Encoder
    inputs = keras.Input(shape=input_shape)
    c1 = conv_block(inputs, 64)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, 128)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = conv_block(p2, 256)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # Middle
    middle = conv_block(p3, 512)

    # Decoder
    up1 = layers.UpSampling2D((2, 2))(middle)
    c4 = conv_block(up1, 256)

    up2 = layers.UpSampling2D((2, 2))(c4)
    c5 = conv_block(up2, 128)

    up3 = layers.UpSampling2D((2, 2))(c5)
    c6 = conv_block(up3, 64)

    # Output
    # output1 = layers.Conv2D(num_classes, (1, 1), activation='softmax', name='output1')(c4)
    # output2 = layers.Conv2D(num_classes, (1, 1), activation='softmax', name='output2')(c5)
    output3 = layers.Conv2D(num_classes, (1, 1), activation='softmax', name='output3')(c6)

    # model = keras.Model(inputs=inputs, outputs=[output1, output2, output3])
    model = keras.Model(inputs=inputs, outputs=output3)

    return model


def Model_Unet3plus(Data, target):
    # Example usage
    input_shape = (256, 256, 3)  # Adjust input shape as needed
    num_classes = 3  # Modify the number of classes
    IMG_SIZE = [256, 256, 3]
    Train_Temp = np.zeros((Data.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
    for i in range(Data.shape[0]):
        Train_Temp[i, :] = np.resize(Data[i], (IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
    Data_X = Train_Temp.reshape(Train_Temp.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])

    Test_Temp = np.zeros((target.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
    for i in range(target.shape[0]):
        Test_Temp[i, :] = np.resize(target[i], (IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
    Data_Y = Test_Temp.reshape(Test_Temp.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])

    tar = []
    for i in range(target.shape[0]):
        targ = cv2.cvtColor(target[i], cv2.COLOR_GRAY2RGB)
        tar.append(targ)
    tar = np.asarray(tar)

    model = unet_3plus(input_shape, num_classes)
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(Data_X, Data_Y, epochs=250)  # , validation_data=(Data, target))
    pred = model.predict(Data)
    return pred
