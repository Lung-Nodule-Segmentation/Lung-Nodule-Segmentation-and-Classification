import numpy as np
from keras.src.models import Model
from keras.src.layers import Conv2D, MaxPooling2D, Dense, Input, Activation, Dropout, GlobalAveragePooling2D, \
    BatchNormalization, concatenate, AveragePooling2D
from sklearn.model_selection import train_test_split
from keras import backend as K
from Model_LSTM import Model_LSTM


def conv_layer(conv_x, filters):
    conv_x = BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)
    conv_x = Conv2D(filters, (3, 3), kernel_initializer='he_uniform', padding='same', use_bias=False)(conv_x)
    conv_x = Dropout(0.2)(conv_x)

    return conv_x


def dense_block(block_x, filters, growth_rate, layers_in_block):
    for i in range(layers_in_block):
        each_layer = conv_layer(block_x, growth_rate)
        block_x = concatenate([block_x, each_layer], axis=-1)
        filters += growth_rate

    return block_x, filters


def transition_block(trans_x, tran_filters):
    trans_x = BatchNormalization()(trans_x)
    trans_x = Activation('relu')(trans_x)
    trans_x = Conv2D(tran_filters, (1, 1), kernel_initializer='he_uniform', padding='same', use_bias=False)(trans_x)
    trans_x = AveragePooling2D((2, 2), strides=(2, 2))(trans_x)

    return trans_x, tran_filters


def dense_net(num_of_class=1, sol=None):
    if sol is None:
        sol = [5, 5, 5]
    dense_block_size = 3
    layers_in_block = 4
    growth_rate = 12
    filters = growth_rate * 2
    input_img = Input(shape=(32, 32, 3))
    x = Conv2D(24, (3, 3), kernel_initializer='he_uniform', padding='same', use_bias=False)(input_img)

    dense_x = BatchNormalization()(x)
    dense_x = Activation('relu')(x)

    dense_x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(dense_x)
    for block in range(dense_block_size - 1):
        dense_x, filters = dense_block(dense_x, filters, growth_rate, layers_in_block)
        dense_x, filters = transition_block(dense_x, filters)

    dense_x, filters = dense_block(dense_x, filters, growth_rate, layers_in_block)
    dense_x = BatchNormalization()(dense_x)
    dense_x = Activation('relu')(dense_x)
    dense_x = GlobalAveragePooling2D()(dense_x)

    dense_y = Dense(int(sol[0]), activation='softmax')(dense_x)
    output = Dense(num_of_class, activation='softmax')(dense_y)
    model = Model(input_img, output)

    return model


def Model_DenseNet(Train_Data, Train_Target, test_data, test_tar, Batch_size=None, sol=None):
    if Batch_size is None:
        Batch_size = 4
    if sol is None:
        sol = [5, 5]
    IMG_SIZE = [32, 32, 3]
    Train_Temp = np.zeros((Train_Data.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
    for i in range(Train_Data.shape[0]):
        Train_Temp[i, :] = np.resize(Train_Data[i], (IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
    Data_X = Train_Temp.reshape(Train_Temp.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])

    Test_Temp = np.zeros((test_data.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
    for i in range(test_data.shape[0]):
        Test_Temp[i, :] = np.resize(test_data[i], (IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
    Data_Y = Test_Temp.reshape(Test_Temp.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])

    model = dense_net(Train_Target.shape[1], sol)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(Data_X, Train_Target, steps_per_epoch=10, batch_size=Batch_size, epochs=int(sol[1]),
              validation_data=(Data_Y, test_tar))
    # pred = model.predict(Data_Y)

    inp = model.input  # input placeholder
    outputs = [layer.output for layer in model.layers]  # all layer outputs
    functors = [K.function([inp], [out]) for out in outputs]  # evaluation functions

    layerNo = -1
    data = np.append(Data_X, Data_Y, axis=0)
    Feats = []
    for i in range(data.shape[0]):
        # print(i, data.shape[0])
        test = data[i, :, :][np.newaxis, ...]
        layer_out = np.asarray(functors[layerNo]([test])).squeeze()  # [func([test]) for func in functors]
        Feats.append(layer_out)
    Feats = np.asarray(Feats)
    Feature = np.resize(Feats, (data.shape[0], 100))
    return Feature


def Model_ADNet_DenseNet(Data, Target, Batchsize=None, sol=None):
    if sol is None:
        sol = [5, 5, 5, 5]
    if Batchsize is None:
        Batchsize = 4

    per = (np.round(Data.shape[0] * 0.75)).astype('int')
    trainX = Data[:per, :]
    trainY = Target[:per, :]
    testX = Data[per:, :]
    testY = Target[per:, :]

    Feature = Model_DenseNet(trainX, trainY, testX, testY, Batchsize, sol)
    y = np.concatenate((trainY, testY), axis=0)
    X_trainFeat, X_testFeat, y_trainFeat, y_testFeat = train_test_split(Feature, y, test_size=0.2, random_state=42)
    EVAL, pred = Model_LSTM(X_trainFeat, y_trainFeat, X_testFeat, y_testFeat, Batchsize, sol)
    return EVAL, pred
