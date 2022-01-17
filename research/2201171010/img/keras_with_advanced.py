# keras frame work

from distutils.log import Log
from shap.datasets import adult #dataset

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

## 1. EDA

X, Y = adult()
print(X.head())
print(X.isnull().sum())
print(X.describe)

## 2. Feature Engineering

numerical_feature=list()
categorical_feature=list()

for (col, typ) in zip(X.dtypes.index, X.dtypes.values):
    if (typ == "float32"):
        numerical_feature.append(col)
    else:
        categorical_feature.append(col)
        
print("numerical: ", numerical_feature)
print("categorical: ",categorical_feature)
print('-----------------------------------------')

# normalization
for column in numerical_feature:
    scaler  = StandardScaler()
    scaler.fit(X[column].values.reshape(-1,1))
    X[column]  = scaler.transform(X[column].values.reshape(-1,1))
    
for column in categorical_feature:
    X[column]  = X[column].astype('category')
    

print(X.dtypes)
    
# one-hot encoding
X = pd.get_dummies(X) # categorical data makes 'one-hot encoding'

x = X.values
y = Y.astype(float) # boolean: {false, true} --> float: (0., 1.0)


## 3. Modeling
from sklearn.model_selection import train_test_split

X_, X_test, y_, y_test = train_test_split(x, y, test_size = 0.1 , stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_, y_, test_size = 0.1 , stratify=y_)


# using LogisticRegression
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(X_train, y_train)
print("Test LR Acc is {}".format(LR.score(X_test, y_test)))


# using DNN with Advanced Techniques

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, ReLU
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.optimizers import Adam

# 2-layer NN (# of hidden nodes: 50)
def scheduler(epoch, lr):
    if epoch in [5, 10, 15]:
        lr = 0.1*lr
    return lr

es_callback = EarlyStopping(monitor='val_acc', patience=5)
lrs_callback = LearningRateScheduler(scheduler)

Hidden_2layer_MLP = Sequential()
Hidden_2layer_MLP.add(Dense(50, 
                            kernel_regularizer=l2(0.01), bias_regularizer=l1(0.01),
                            kernel_initializer= 'he_normal' ))
Hidden_2layer_MLP.add(BatchNormalization())
Hidden_2layer_MLP.add(ReLU())
Hidden_2layer_MLP.add(Dropout(0.5))
Hidden_2layer_MLP.add(Dense(1, activation='sigmoid',
                            kernel_regularizer=l2(0.01), bias_regularizer=l1(0.01),
                            kernel_initializer= 'glorot_normal'))


Hidden_2layer_MLP.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
history = Hidden_2layer_MLP.fit(X_train, y_train, epochs=20, batch_size=128, shuffle=True, callbacks=[es_callback, lrs_callback], validation_data = (X_val, y_val))

## plotting results
import matplotlib.pyplot as plt

train_loss = history.history['loss']
train_acc = history.history['acc']
val_loss = history.history['val_loss']
val_acc = history.history['val_acc']
plt.plot(train_loss)
plt.plot(val_loss)
plt.grid()
plt.title("Loss")
plt.xlabel("epochs")
plt.ylabel("Loss")
plt.legend(['train','val'])

plt.show()

## 학습 모델 저장하고 불러오기
from tensorflow.keras.models import load_model
Hidden_2layer_MLP.save('./model_dnn')
model_load = load_model("./model_dnn")
model_load.evaluate(X_test, y_test)