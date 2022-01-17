#  프레임워크 불러오기
import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Input, ReLU, Softmax

## 데이터 불러오기
from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()


## 모델 설계
inputs = Input(shape=(28,28))
flat= Flatten()(inputs)
dense1 = Dense(128)(flat)
relu1 = ReLU()(dense1)
dense2 = Dense(128)(relu1)
relu2 = ReLU()(dense2)
dense3 = Dense(64)(relu2)
relu3 = ReLU()(dense3)
dense4 = Dense(64)(relu3)
relu4 = ReLU()(dense4)
dense5 = Dense(10)(relu4)
outputs = Softmax()(dense5)
model = Model(inputs=inputs, outputs=outputs)

## 또는, 이렇게 모델 설계도 가능함
'''
model = Sequential()
model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(10, activation='softmax'))
'''

model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics= ['acc'])
model.fit(X_train, y_train, epochs= 20, batch_size = 5000, shuffle=True)

print(model.summary())

test_loss, test_acc = model.evaluate(X_test[:100], y_test[:100])
print("Test accuracy is {}".format(test_acc))