from tensorflow.keras.datasets import cifar10
import tensorflow.keras as keras

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print("input data shape is {}".format(X_train.shape))

# Normalization
X_train, X_test = X_train/255.0 , X_test/255.0

import matplotlib.pyplot as plt
plt.imshow(X_train[100])
plt.show()

# Simplified AlexNet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPool2D

model_alexnet = Sequential()
# 1st Layer: [conv] + [Relu] + [pool]
model_alexnet.add(Conv2D(input_shape=(32,32,3),
                         kernel_size=(3,3),
                         strides=(1,1),
                         filters=48,
                         padding='same',
                         activation='relu'))

model_alexnet.add(MaxPool2D(pool_size=(2,2),
                            strides=(2,2),
                            padding='same'))


# 2nd Layer : [conv] + [Relu] + [pool]
model_alexnet.add(Conv2D(kernel_size=(3,3),
                         strides=(1,1),
                         filters=96,
                         padding='same',
                         activation='relu'))

model_alexnet.add(MaxPool2D(pool_size=(2,2),
                            strides=(2,2),
                            padding='same'))

# 3st Layer : [conv] + [Relu] 

model_alexnet.add(Conv2D(kernel_size=(3,3),
                         strides=(1,1),
                         filters=192,
                         padding='same',
                         activation='relu'))

# 4st Layer : [conv] + [Relu] 

model_alexnet.add(Conv2D(kernel_size=(3,3),
                         strides=(1,1),
                         filters=192,
                         padding='same',
                         activation='relu'))

# 5th Layer : [conv] + [Relu] + [Pool] 

model_alexnet.add(Conv2D(kernel_size=(3,3),
                         strides=(1,1),
                         filters=256,
                         padding='same',
                         activation='relu'))

model_alexnet.add(MaxPool2D(pool_size=(2,2),
                            strides=(2,2),
                            padding='same'))

model_alexnet.add(Flatten())
model_alexnet.add(Dense(512, activation='relu'))
model_alexnet.add(Dense(256, activation='relu'))
model_alexnet.add(Dense(10, activation='softmax'))


model_alexnet.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
model_alexnet.fit(X_train, y_train, epochs=5, batch_size=64)

## ResNet (skip connection을 설계하기 위해, Model 클래스 활용)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, ReLU, Softmax, Add, AveragePooling2D, MaxPool2D, BatchNormalization, Conv2D ,GlobalAveragePooling2D

inputs = Input(shape=(32,32,3))
x = Conv2D(filters= 32, kernel_size=(3,3), strides=(1,1), padding='same', use_bias=False)(inputs)
x = BatchNormalization()(x)
x = ReLU()(x)

x = Conv2D(filters= 32, kernel_size=(3,3), strides=(1,1), padding='same', use_bias=False)(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid')(x)

x = Conv2D(filters= 64,kernel_size = (3,3), strides= (1,1), padding='same', use_bias=False)(x)
x = BatchNormalization()(x)
x=ReLU()(x)
skip = x

x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', use_bias=False)(x)
x = BatchNormalization()(x)
x = ReLU()(x)

x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', use_bias=False)(x)
x = BatchNormalization()(x)
x = ReLU()(x)

x = Add()([x,skip])
x = ReLU()(x)

x= MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid')(x)
x= Conv2D(128, kernel_size=(3,3), strides=(1,1), padding='same', use_bias=False)(x)
x= BatchNormalization()(x)
x= ReLU()(x)
skip2 = x

x = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same', use_bias=False)(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same', use_bias=False)(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = Add()([skip2, x])
x=AveragePooling2D(pool_size=(8,8), strides=(1,1), padding='valid')(x)

outputs=Dense(10, activation='relu')(x)

model_resnet = Model(inputs=inputs, outputs=outputs)
print(model_resnet.summary())

model_resnet.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
model_resnet.fit(X_train, y_train, epochs=20, batch_size=64)