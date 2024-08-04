from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 데이터 가져오기
x, y = load_iris(return_X_y=True)
# Train, Test 데이터 나누기
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2)


# 모델링하기
from sklearn.neighbors import KNeighborsClassifier
#print(help(KNeighborsClassifier)) 로 함수 사용법을 알 수 있다.

KNC = KNeighborsClassifier(n_neighbors=3)
KNC.fit(X_train, y_train)
knc_score = KNC.score(X_test, y_test)

print("KNC score is {}".format(knc_score))