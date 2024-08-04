from typing import Sequence
import tensorflow as tf

# CNN using Keras Practice
from tensorflow.keras.datasets import mnist

# Load Data
(X,Y), (X_test, Y_test) = mnist.load_data()

# Normalize image data as 255.0
X = X/255.0
X_test = X_test / 255.0


# LeNet-5
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import BatchNormalization, Dropout, Dense, Flatten, ReLU, Softmax
from tensorflow.keras.layers import Conv2D, MaxPool2D

X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

model_lenet = Sequential()
# 1st stack
model_lenet.add(Conv2D(input_shape=(28,28,1),
                       kernel_size = (5,5),
                       strides=(1,1),
                       filters=32,
                       padding='same',
                       activation='relu')
                )
model_lenet.add(MaxPool2D(pool_size=(2,2),
                          strides=(2,2),
                          padding= 'valid')
                )
# 2nd stack
model_lenet.add(Conv2D(kernel_size=(5,5),
                       strides=(1,1),
                       filters=48,
                       padding='same',
                       activation='relu'))
model_lenet.add(MaxPool2D(pool_size=(2,2), strides=(2,2),
                          padding='valid'))
# Fully Connected Layer
model_lenet.add(Flatten())
model_lenet.add(Dense(256, activation='relu'))
model_lenet.add(Dense(84, activation='relu'))
model_lenet.add(Dense(10, activation='softmax'))

model_lenet.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
print(model_lenet.summary())

## 학습하기
model_lenet.fit(X, Y, epochs=10, batch_size=128)
lenet_score = model_lenet.evaluate(X, Y, batch_size=128)
print(lenet_score)