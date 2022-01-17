# Practice #1
from distutils.log import Log
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Load the data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("X_train shape is {}".format(X_train.shape))
print("y_train shape is {}".format(y_train.shape))
#plt.imshow(X_test[3])

# Normalize the dataset
X_train = X_train/255.0
X_test = X_test/255.0


## Modeling and Training(1): LogisticRegression
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression()
lr_model.fit(X=X_train.reshape(X_train.shape[0], -1), y=y_train)
pred=lr_model.predict(X_test.reshape(X_test.shape[0], -1))

print("X_test[3] is predicted as {}".format(pred[3]))
lr_score = lr_model.score(X_test.reshape(X_test.shape[0], -1), y_test)
print("Logistic Regression Score is {}".format(lr_score))
#plt.show()


## Modeling and Training(2): RidgeClassifier
# - Weights 를 L2 Normalization 한 것을 Loss 함수에 추가한 것을 Ridge라고 부른다.
# - Weights 를 L1 Normalization 한 것을 Loss 함수에 추가한 것을 Lasso라고 한다.

from sklearn.linear_model import RidgeClassifier
rc_model = RidgeClassifier()
rc_model.fit(X=X_train.reshape(X_train.shape[0], -1), y=y_train)
pred = rc_model.predict(X_test.reshape(X_test.shape[0], -1))

print("X_test[4] is predicted as {}".format(pred[4]))
rc_score = rc_model.score(X=X_test.reshape(X_test.shape[0], -1), y=y_test)
print("Ridge Classifier Score is {}".format(rc_score))
#plt.imshow(X_test[4])
#plt.show()


## 이미지데이터는 픽셀이니까 /255.0 으로 Normalization 이 가능하지만, 그렇지 않은 UserData인 경우도 있다.
## Data 전처리 함수를 이용해서 전처리도 가능하다.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(X_train.reshape(X_train.shape[0], -1))
X_train_scaled = scaler.transform(X_train.reshape(X_train.shape[0], -1))