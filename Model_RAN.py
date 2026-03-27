import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import numpy as np
from Evaluation import evaluation


# Residual Block
def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, kernel_size, strides=1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x


# Attention Module
def attention_module(x, filters):
    residual = x
    # Attention mechanism
    attention = layers.Conv2D(filters, kernel_size=3, padding="same", activation="sigmoid")(x)
    x = layers.Multiply()([x, attention])
    x = layers.Add()([x, residual])  # Residual connection
    return x


# Residual Attention Network (RAN)
def RAN(input_shape, num_classes, HN):
    inputs = Input(shape=input_shape)

    # Initial Conv Block
    x = layers.Conv2D(64, kernel_size=7, strides=2, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(x)

    # Residual Blocks with Attention
    x = residual_block(x, 64, stride=1)
    x = attention_module(x, 64)

    x = residual_block(x, 128, stride=2)
    x = attention_module(x, 128)

    x = residual_block(x, 256, stride=2)
    x = attention_module(x, 256)

    x = residual_block(x, 512, stride=2)
    x = attention_module(x, 512)

    # Classification Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(HN, activation="relu")(x)  # 512
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    # Create Model
    model = Model(inputs, outputs)
    return model


def Model_RAN(train_data, train_target, test_data, test_target, EP=None, BS=None, HN=None, sol=None):
    if BS is None:
        BS = 4
    if HN is None:
        HN = 512
    if EP is None:
        EP = 10
    if sol is None:
        sol = [5, 1]
    input_shape = (224, 224, 3)
    num_classes = train_target.shape[-1]

    X_train = np.zeros((train_data.shape[0], input_shape[0], input_shape[1], input_shape[2]))
    for i in range(train_data.shape[0]):
        temp = np.resize(train_data[i], (input_shape[0] * input_shape[1], input_shape[2]))
        X_train[i] = np.reshape(temp, (input_shape[0], input_shape[1], input_shape[2]))

    X_test = np.zeros((test_data.shape[0], input_shape[0], input_shape[1], input_shape[2]))
    for i in range(test_data.shape[0]):
        temp = np.resize(test_data[i], (input_shape[0] * input_shape[1], input_shape[2]))
        X_test[i] = np.reshape(temp, (input_shape[0], input_shape[1], input_shape[2]))

    model = RAN(input_shape, num_classes, HN)
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                  metrics=['accuracy'])
    model.summary()
    model.fit(X_train, train_target, steps_per_epoch=10, verbose=1, batch_size=4, epochs=EP,
              validation_data=(X_test, test_target))
    pred = model.predict(X_test)
    avg = np.mean(pred)
    pred[pred >= avg] = 1
    pred[pred < avg] = 0
    pred = pred.astype('int')
    Eval = evaluation(test_target, pred)
    return Eval, pred
