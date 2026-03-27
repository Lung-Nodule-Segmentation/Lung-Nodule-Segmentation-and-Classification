import keras
from keras.src.models import Model
from keras.src.layers import Input, Conv3D, MaxPooling3D, UpSampling3D
from keras.src.layers import concatenate, Dropout, BatchNormalization
from keras.src.layers import LayerNormalization, Dense, Reshape, MultiHeadAttention
import numpy as np


# -------- Transformer Block --------
def transformer_block(x, num_heads=4, embed_dim=64):
    shape = x.shape
    B, D, H, W, C = shape

    x_flat = Reshape((D * H * W, C))(x)

    attn = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x_flat, x_flat)
    x_flat = LayerNormalization()(x_flat + attn)

    ffn = Dense(embed_dim, activation='relu')(x_flat)
    ffn = Dense(C)(ffn)

    x_flat = LayerNormalization()(x_flat + ffn)

    x_out = Reshape((D, H, W, C))(x_flat)
    return x_out


# -------- Dense Block --------
def dense_block(x, filters):
    c1 = Conv3D(filters, 3, padding='same', activation='relu')(x)
    c1 = BatchNormalization()(c1)

    c2 = Conv3D(filters, 3, padding='same', activation='relu')(c1)
    c2 = BatchNormalization()(c2)

    out = concatenate([x, c1, c2])
    return out


# -------- 3D TDUNet++ --------
def tdunetpp_3d(input_size=(64, 64, 64, 1), num_classes=1):

    inputs = Input(input_size)

    # Encoder
    c1 = dense_block(inputs, 32)
    p1 = MaxPooling3D((2,2,2))(c1)

    c2 = dense_block(p1, 64)
    p2 = MaxPooling3D((2,2,2))(c2)

    c3 = dense_block(p2, 128)
    p3 = MaxPooling3D((2,2,2))(c3)

    # Bottleneck + Transformer
    b = dense_block(p3, 256)
    b = transformer_block(b)

    # Decoder (UNet++ style skip fusion)
    u3 = UpSampling3D((2,2,2))(b)
    u3 = concatenate([u3, c3])
    c6 = dense_block(u3, 128)

    u2 = UpSampling3D((2,2,2))(c6)
    u2 = concatenate([u2, c2])
    c7 = dense_block(u2, 64)

    u1 = UpSampling3D((2,2,2))(c7)
    u1 = concatenate([u1, c1])
    c8 = dense_block(u1, 32)

    outputs = Conv3D(num_classes, 1, activation='sigmoid')(c8)

    model = Model(inputs, outputs)
    return model


def Model_3D_Trans_DenseUnet_Plus_plus(Images, GT, sol=None):
    if sol is None:
        sol = [4, 20, 1, 5, 0]
    IMG_SIZE = 64   # 3D size
    classes = 1
    optimizer_list = ['SGD', 'Adam', 'RMSprop', 'Adagrad', 'Adadelta']
    input_shape = (IMG_SIZE, IMG_SIZE, IMG_SIZE, 1)

    # -------- Resize Images --------
    Train_X = np.zeros((Images.shape[0], *input_shape))
    for i in range(Images.shape[0]):
        Train_X[i] = np.resize(Images[i], input_shape)

    Train_Y = np.zeros((GT.shape[0], *input_shape))
    for i in range(GT.shape[0]):
        Train_Y[i] = np.resize(GT[i], input_shape)

    # -------- Build Model --------
    model = tdunetpp_3d(input_size=input_shape, num_classes=classes)
    model.compile(optimizer=optimizer_list[int(sol[2])], loss=keras.losses.binary_crossentropy, metrics=['accuracy'])
    model.summary()
    model.fit(Train_X, Train_Y, epochs=sol[1], steps_per_epoch=2, verbose=1)
    Predict = model.predict(Train_X)
    return Predict
