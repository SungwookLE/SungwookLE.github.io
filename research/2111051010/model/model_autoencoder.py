from tensorflow.keras.layers import Dense, Flatten, Conv2D, ReLU, BatchNormalization, MaxPool2D, Dropout, Input, Softmax, Conv2DTranspose
from tensorflow.keras.models import Model, Sequential

def model_autoencoder(input_shape = (64, 64, 3)):

    #Encoder
    inputs = Input(shape=input_shape)
    conv1 = Conv2D(kernel_size=(5,5), strides=(1,1), filters=64, padding='same', kernel_initializer='he_normal', name='enc_1st_conv')(inputs)
    conv1 = ReLU(name='enc_1st_relu')(conv1)
    pool1 = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid', name='enc_1st_maxpool')(conv1)
    conv2 = Conv2D(kernel_size=(5,5), strides=(1,1), filters=128, padding='same', kernel_initializer='he_normal', name='enc_2nd_conv')(pool1)
    conv2 = ReLU(name='enc_2nd_relu')(conv2)
    pool2 = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid', name='enc_2nd_maxpool')(conv2)
    conv3 = Conv2D(kernel_size=(5,5), strides=(1,1), filters=256, padding='same', kernel_initializer='he_normal', name='enc_3rd_conv')(pool2)
    conv3 = ReLU(name='enc_3rd_relu')(conv3)

    latent = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid', name='enc_3rd_maxpool')(conv3)

    

    #Decoder
    dec = Conv2DTranspose(kernel_size=(5,5), strides=(2,2), filters= 256, padding = 'same', kernel_initializer='he_normal', name='dec_3rd_conv', activation='relu')(latent)
    dec = Conv2DTranspose(kernel_size=(5,5), strides=(2,2), filters= 128, padding = 'same', kernel_initializer='he_normal', name='dec_2nd_conv', activation='relu')(dec)
    dec = Conv2DTranspose(kernel_size=(5,5), strides=(2,2), filters= 64, padding = 'same', kernel_initializer='he_normal', name='dec_1st_conv', activation='relu')(dec)
    
    out = Conv2D(kernel_size=(5,5), strides=(1,1), filters= input_shape[2], padding = 'same', kernel_initializer='he_normal', name='dec_out', activation='relu')(dec)

    entire_model = Model(inputs= inputs, outputs = out)
    return entire_model

def model_classifier_with_encoder(encoder, input_shape=(64,64,3), train_classifier=None):

    encoder.trainable= False

    inputs = Input(shape=input_shape, name='input_layer')
    feature = encoder(inputs)

    # conv_fin = Conv2D(kernel_size=(5,5), strides=(1,1), filters=256, padding='same', kernel_initializer='he_normal', name='adhesive_conv')(feature)
    # conv_fin = ReLU(name='adhesive_relu')(conv_fin)
    # pool_fin = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid', name='adhesive_maxpool')(conv_fin)

    dropout = Dropout(0.5)(feature)
    flat = Flatten()(dropout)
    
    # 멀티 아웃풋 헤드
    if (train_classifier=="OOP"):
        fc1 = Dense(256, activation='relu', kernel_initializer='he_normal', name='fc1')(flat)
        dropout1 = Dropout(0.5)(fc1)
        out_oop= Dense(5, activation='softmax', kernel_initializer='he_normal', name='out_oop')(dropout1)
        entire_model = Model(inputs=inputs, outputs=out_oop, name='enc_oop')

    elif (train_classifier=="Weak"):
        fc2 = Dense(128, activation='relu', kernel_initializer='he_normal', name='fc2')(flat)
        dropout2 = Dropout(0.5)(fc2)
        out_weak= Dense(2, activation='softmax', kernel_initializer='he_normal', name='out_weak')(dropout2)
        entire_model = Model(inputs=inputs, outputs=out_weak, name='enc_weak')

    elif (train_classifier=="Mask"):
        fc3 = Dense(128, activation='relu', kernel_initializer='he_normal', name='fc3')(flat)
        dropout3 = Dropout(0.5)(fc3)
        out_mask= Dense(2, activation='softmax', kernel_initializer='he_normal', name='out_mask')(dropout3)
        entire_model = Model(inputs=inputs, outputs=out_mask, name='enc_mask')

    return entire_model


def multihead_classifier_with_encoder(encoder, oop, weak, mask, input_shape=(64,64,3)):

    '''
    Final Model, fin
    '''
    encoder.trainable= False
    oop.trainable= False
    weak.trainable= False
    mask.trainable= False

    inputs = Input(shape=input_shape, name='input_layer')
    feature = encoder(inputs)

    dropout = Dropout(0.5)(feature)
    flat = Flatten()(dropout)
    
    # 멀티 아웃풋 헤드
    out_oop = oop(flat)
    out_weak= weak(flat)
    out_mask= mask(flat)
    entire_model = Model(inputs=inputs, outputs=[out_oop, out_weak, out_mask], name='final_multi')

    return entire_model
