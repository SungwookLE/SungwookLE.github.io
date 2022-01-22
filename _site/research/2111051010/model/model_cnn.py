from tensorflow.keras.layers import Dense, Flatten, Conv2D, ReLU, BatchNormalization, MaxPool2D, Dropout, Input, Softmax
from tensorflow.keras.models import Model, Sequential

def model_cnn(input_shape = (128, 128, 3), classifier = None):

    inputs = Input(shape=input_shape)
    #base_conv = base_conv_net(input_shape=input_shape, base_trainable= base_trainable)(inputs)

    conv1 = Conv2D(kernel_size=(5,5), strides=(1,1), filters=64, padding='same', kernel_initializer='he_normal', name='1st_conv')(inputs)
    conv1 = ReLU(name='1st_relu')(conv1)
    pool1 = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid', name='1st_maxpool')(conv1)


    conv2 = Conv2D(kernel_size=(5,5), strides=(1,1), filters=128, padding='same', kernel_initializer='he_normal', name='2nd_conv')(pool1)
    # conv2 = BatchNormalization()(conv2)
    conv2 = ReLU(name='2nd_relu')(conv2)
    # conv2 = Dropout(0.5)(conv2)
    pool2 = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid', name='2nd_maxpool')(conv2)

    conv3 = Conv2D(kernel_size=(5,5), strides=(1,1), filters=256, padding='same', kernel_initializer='he_normal', name='3rd_conv')(pool2)
    conv3 = ReLU(name='3rd_relu')(conv3)
    pool3 = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid', name='3rd_maxpool')(conv3)

    dropout = Dropout(0.5)(pool3)
    flat = Flatten()(dropout)

    fc1 = Dense(128, activation='relu', kernel_initializer='he_normal', name='fc1')(flat)
    dropout = Dropout(0.5)(fc1)
    
    # 멀티 아웃풋 헤드
    if (classifier=="Belt"):
        out_belt= Dense(2, activation='softmax', kernel_initializer='he_normal', name='out_belt')(dropout)
        entire_model = Model(inputs=inputs, outputs=out_belt, name='belt_classifier')

    elif (classifier=="Weak"):
        out_weak= Dense(2, activation='softmax', kernel_initializer='he_normal', name='out_weak')(dropout)
        entire_model = Model(inputs=inputs, outputs=out_weak, name='weak_classifier')
    
    elif (classifier=="OOP"):
        out_oop= Dense(5, kernel_initializer='he_normal', name='out_oop', activation='softmax')(dropout)
        entire_model = Model(inputs=inputs, outputs=out_oop, name='oop_classifier')

    elif (classifier=="Mask"):
        out_mask= Dense(2, activation='softmax', kernel_initializer='he_normal', name='out_mask')(dropout)
        entire_model = Model(inputs=inputs, outputs=out_mask, name='mask_classifier')

    return entire_model