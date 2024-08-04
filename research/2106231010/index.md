---
layout: post
type: research
date: 2021-06-23 10:10
category: Kaggle
title: DataAnalysis Kaggle Titanic by Myself
subtitle: Classifier- Classification with feature engineering
writer: 100
post-header: true
header-img: https://static1.squarespace.com/static/5006453fe4b09ef2252ba068/t/5090b249e4b047ba54dfd258/1351660113175/TItanic-Survival-Infographic.jpg?format=1500w
hash-tag: [Kaggle, Classifier]
use_math: true
---

#  Data Analysis: Kaggle Titanic `Survived` Predict  
AUTHOR: SungwookLE  
DATE: '21.6/23  
PROBLEM: Classifier [Kaggle LINK](https://www.kaggle.com/c/titanic)  
REFERENCE:  
- [#1 LECTURE](https://www.youtube.com/watch?v=aqp_9HV58Ls&list=RDCMUCxP77kNgVfiiG6CXZ5WMuAQ&index=3)  
- [#2 LECTURE](https://www.youtube.com/watch?v=nXFXAxfdIls&list=PLVNY1HnUlO25B-8Gwn1mS35SD0yMHh147&index=3)  
- [#3 LECTURE](https://www.youtube.com/watch?v=FAP7JOECfEE&list=RDCMUCxP77kNgVfiiG6CXZ5WMuAQ&index=1)  

- **The Challenge**  
In this challenge, we ask you to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).

- **Given Data**  
The data has been split into two groups:  
- training set (train.csv)
- test set (test.csv)  

- **Data Dictionary**  

    ||||
    |:--|:--|:--|
    |Variable|Definition|Key|
    |survival|Survival|0 = No, 1 = Yes|
    |pclass|Ticket class|1 = 1st, 2 = 2nd, 3 = 3rd|
    |sex|Sex||
    |Age|Age in years||	
    |sibsp|# of siblings / spouses aboard the Titanic||
    |parch|# of parents / children aboard the Titanic||
    |ticket|Ticket number||
    |fare|Passenger fare||
    |cabin|Cabin number||
    |embarked|Port of Embarkation|C = Cherbourg, Q = Queenstown, S = Southampton|

- **Variable Notes**  
    - pclass: A proxy for socio-economic status (SES)  
    1st = Upper  
    2nd = Middle  
    3rd = Lower  
    - age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5  
    - sibsp: The dataset defines family relations in this way...  
    - Sibling = brother, sister, stepbrother, stepsister  
    - Spouse = husband, wife (mistresses and fiancés were ignored)  
    - parch: The dataset defines family relations in this way...  
    - Parent = mother, father  
    - Child = daughter, son, stepdaughter, stepson Some children travelled only with a nanny, therefore parch=0 for them.  

![image](https://static1.squarespace.com/static/5006453fe4b09ef2252ba068/t/5090b249e4b047ba54dfd258/1351660113175/TItanic-Survival-Infographic.jpg?format=1500w)

### OVERVIEW
1) Data Analysis
- 데이터 차원, 형태 파악하기
- 그래프 그려서 에측변수 `SalePrice`와 다른 변수와의 상관관계 파악하기

2) Feature Engineering  
2-1) categorical + numerical features 분리하기
- using `select_dtypes()`.    
- numerical 데이터 중 month나 year 등의 데이터는 categorical로 분류해주기 `apply(str)`  
      
2-2) 비어있는 missing 데이터 채우기  
- numerical: mean, median, mode 를 활용하여 데이터 채우기 `.fillna(xxx)`, `mean(), median(), mode()`
- categorical: `pd.get_dummies()` 나 `LabelEncoder`를 활용해서 missing 데이터도 없애고, one-hot encoding도 완성하기

2-3) data의 skewness 줄이기 
- numerical data 의 skewness 줄이기  

2-4) new feature / del feature
- 필요하다면

3) Modeling
- CrossValidation using `cross_val_score, KFold. train_test_split`.          
- Regressor : `LinearRegression, RidgeCV, LassoCV, ElasticNetCV` 
- Classifier  : `KNN, RandomForest, ...`
- Techniques: `StandardScaler, RobustScaler`.
- Easy modeling: `make_pipeline`

## START

## 1. Data Analysis
- 데이터 차원, 형태 파악하기
- 그래프 그려서 에측변수 `SalePrice`와 다른 변수와의 상관관계 파악하기


```python
from subprocess import check_output
import pandas as pd

print(check_output(["ls","input"]).decode('utf8'))
```

    test.csv
    train.csv
    



```python
train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')
```


```python
print("Initial train data shape is {}".format(train.shape))
n_train = train.shape[0]
train.head(3)
```

    Initial train data shape is (891, 12)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
print("Initial test data shape is {}".format(test.shape))
n_test = test.shape[0]
test.head(3)
```

    Initial test data shape is (418, 11)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>3</td>
      <td>Kelly, Mr. James</td>
      <td>male</td>
      <td>34.5</td>
      <td>0</td>
      <td>0</td>
      <td>330911</td>
      <td>7.8292</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>3</td>
      <td>Wilkes, Mrs. James (Ellen Needs)</td>
      <td>female</td>
      <td>47.0</td>
      <td>1</td>
      <td>0</td>
      <td>363272</td>
      <td>7.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>2</td>
      <td>Myles, Mr. Thomas Francis</td>
      <td>male</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>240276</td>
      <td>9.6875</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
  </tbody>
</table>
</div>




```python
all_data = pd.concat([train,test],axis=0).reset_index(drop=True)
```


```python
unique_id = len(set(all_data['PassengerId']))
total_count = len(all_data)

diff = unique_id-total_count
print("Difference with unique-Id and total Count: {}".format(diff))
```

    Difference with unique-Id and total Count: 0



```python
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()
```


```python
y_label = all_data['Survived'][:n_train]
all_data.drop('PassengerId', axis=1, inplace=True)
all_data.drop('Survived',axis=1, inplace=True)

def bar_chart(feature):
    survived = train.loc[train['Survived']==1, feature].value_counts()
    dead = train.loc[train['Survived']==0, feature].value_counts()

    df = pd.DataFrame([survived, dead], index=['Survived', 'Dead'])
    df.plot(kind='bar', stacked=True, figsize=(10,5),title=("Survived with "+feature))
```


```python
bar_chart('Sex')
```


    
![svg](/assets/titanic_my_files/titanic_my_12_0.svg)
    



```python
corrmat= train.corr()
f, ax = plt.subplots(figsize=(10,8))
sns.heatmap(corrmat, vmax=0.8, annot=True)

abs(corrmat['Survived']).sort_values(ascending =False)
```




    Survived       1.000000
    Pclass         0.338481
    Fare           0.257307
    Parch          0.081629
    Age            0.077221
    SibSp          0.035322
    PassengerId    0.005007
    Name: Survived, dtype: float64




    
![svg](/assets/titanic_my_files/titanic_my_13_1.svg)
    



```python
def facet_plot(feature, range_opt=None):
    facet = sns.FacetGrid(train, hue='Survived', aspect=4)
    facet.map(sns.kdeplot, feature, shade = True)

    if not range_opt:
        facet.set(xlim=(0, train[feature].max()))
    else:
        facet.set(xlim=range_opt)
    facet.add_legend()
    plt.title("Survived with "+feature)
    plt.show()
```


```python
facet_plot('Age')
```


    
![svg](/assets/titanic_my_files/titanic_my_15_0.svg)
    



```python
all_data.isnull().sum()
```




    Pclass         0
    Name           0
    Sex            0
    Age          263
    SibSp          0
    Parch          0
    Ticket         0
    Fare           1
    Cabin       1014
    Embarked       2
    dtype: int64




```python
all_data.dtypes.value_counts()
```




    object     5
    int64      3
    float64    2
    dtype: int64




```python
print("Train Y Label Data is {}".format(y_label.shape))
print("All Data is {}".format(all_data.shape))
```

    Train Y Label Data is (891,)
    All Data is (1309, 10)




## 2. Feature Engineering
### 2-1. Categorical + numerical features 분리하기
- using `select_dtypes()`
- numerical 데이터 중 month나 year 등의 데이터는 categorical로 분류해주기 `apply(str)`


```python
all_data['Pclass']=all_data['Pclass'].apply(str)
print("Numerical Feature is {}".format(len(all_data.select_dtypes(exclude=object).columns)))
numerical_features = all_data.select_dtypes(exclude=object).columns
numerical_features
```

    Numerical Feature is 4





    Index(['Age', 'SibSp', 'Parch', 'Fare'], dtype='object')




```python
print("Categorical Feature is {}".format(len(all_data.select_dtypes(include=object).columns)))
categorical_features = all_data.select_dtypes(include=object).columns
categorical_features
```

    Categorical Feature is 6





    Index(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], dtype='object')



### 2-2. 비어있는 missing 데이터 채우기  
- numerical: mean, median, mode 를 활용하여 데이터 채우기 `.fillna(xxx)`, `mean(), median(), mode()`
- categorical: `pd.get_dummies()` 나 `LabelEncoder`를 활용해서 missing 데이터도 없애고, one-hot encoding도 완성하기


```python
all_data.isnull().sum()
```




    Pclass         0
    Name           0
    Sex            0
    Age          263
    SibSp          0
    Parch          0
    Ticket         0
    Fare           1
    Cabin       1014
    Embarked       2
    dtype: int64




```python
# 숫자 데이터
all_data['Age'].fillna(all_data.groupby('Sex')['Age'].transform('median'), inplace=True)
```


```python
# 숫자 데이터
all_data['Fare'].fillna(all_data.groupby('Pclass')['Fare'].transform('median'), inplace=True)
```


```python
all_data.drop('Cabin',axis=1, inplace=True)
all_data.drop('Ticket',axis=1, inplace=True)
categorical_features=categorical_features.drop('Cabin')
categorical_features=categorical_features.drop('Ticket')
```


```python
# 카테고리칼 데이터
all_data['Embarked'].fillna(all_data['Embarked'].value_counts().sort_values(ascending=False).index[0],inplace=True)
```


```python
all_data['Name']=all_data['Name'].str.extract('([A-Za-z]+)\.', expand=False)
```


```python
all_data.isnull().sum()
```




    Pclass      0
    Name        0
    Sex         0
    Age         0
    SibSp       0
    Parch       0
    Fare        0
    Embarked    0
    dtype: int64




```python
print("Missing(NA) Data is {}".format(all_data.isnull().values.sum()))
```

    Missing(NA) Data is 0


`pd.get_dummies`로 categorical데이터 one-hot encoding 해주기


```python
all_data=pd.get_dummies(all_data)
```


```python
print("After fill and ONE-HOT encoding data shape is {}".format(all_data.shape))
```

    After fill and ONE-HOT encoding data shape is (1309, 30)


### 2-3. data의 skewness 줄이기 
- y_label 데이터도 skewness 가 있으면 줄인다음에 학습하는 것이 학습결과에 이득: classifier 문제에서는 skewness를 확인할 수는 없지
- numerical data 의 skewness 줄이기  


```python
from scipy import stats
from scipy.stats import norm, skew # for some statistics

skewness = all_data[numerical_features].apply(lambda x: skew(x.dropna()))
skewness = skewness.sort_values(ascending=False)
skewness_features = skewness[abs(skewness.values)>1].index
print("skewness:")
print(skewness_features)

plt.figure(figsize=(10,5))
plt.xticks(rotation='90')
sns.barplot(x=skewness.index, y=skewness.values)
plt.title('Before skewness elimination using log1p')
```

    skewness:
    Index(['Fare', 'SibSp', 'Parch'], dtype='object')





    Text(0.5, 1.0, 'Before skewness elimination using log1p')




    
![svg](/assets/titanic_my_files/titanic_my_36_2.svg)
    



```python
sns.distplot(all_data['Fare'], fit=norm)

(mu,sigma) = norm.fit(all_data['Fare'])
plt.legend(['Normal dist. mu={:.2f}, std={:.2f}'.format(mu,sigma)], loc='best')
plt.ylabel('Frequency')
plt.title('Before skewness in Fare')

fig = plt.figure()
res = stats.probplot(all_data['Fare'], plot=plt)
plt.show()
```


    
![svg](/assets/titanic_my_files/titanic_my_37_0.svg)
    



    
![svg](/assets/titanic_my_files/titanic_my_37_1.svg)
    



```python
import numpy as np

#Fare, SibSp, Parch

for col in skewness_features:
    all_data[col] = np.log1p(all_data[col])
```


```python
skewness = all_data[numerical_features].apply(lambda x: skew(x.dropna()))
skewness = skewness.sort_values(ascending=False)
print(skewness)

plt.figure(figsize=(10,5))
plt.xticks(rotation='90')
sns.barplot(x=skewness.index, y=skewness.values)
plt.title('After skewness elimination using log1p')
```

    Parch    1.787711
    SibSp    1.634945
    Age      0.552731
    Fare     0.542519
    dtype: float64





    Text(0.5, 1.0, 'After skewness elimination using log1p')




    
![svg](/assets/titanic_my_files/titanic_my_39_2.svg)
    



```python
sns.distplot(all_data['Fare'], fit=norm)

(mu,sigma) = norm.fit(all_data['Fare'])
plt.legend(['Normal dist. mu={:.2f}, std={:.2f}'.format(mu,sigma)], loc='best')
plt.ylabel('Frequency')
plt.title('After skewness in Fare')


fig = plt.figure()
res = stats.probplot(all_data['Fare'], plot=plt)
plt.show()
```


    
![svg](/assets/titanic_my_files/titanic_my_40_0.svg)
    



    
![svg](/assets/titanic_my_files/titanic_my_40_1.svg)
    


### 2-4. new feature / del feature
- 필요하다면 하는 것이고, 여기선 하지 않겠다.


```python
train_data = all_data[:n_train]
test_data = all_data[n_train:]
```

## 3. Modeling
- CrossValidation using `cross_val_score, KFold. train_test_split`.          
- Regressor : `LinearRegression, RidgeCV, LassoCV, ElasticNetCV` 
- Classifier  :  
    1) kNN (가까운 이웃)  
    2) Decision Tree (논리 순서)  
    3) Random Forest (논리 순서, 여러개 세트를 두고 다수결)  
    4) 베이지안 룰 (확률)  
    5) SVM (서포트 벡터 머신))  

- Techniques: `StandardScaler, RobustScaler`.
- Easy modeling: `make_pipeline`


```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
```


```python
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits = 10, shuffle = True, random_state = 0)
```


```python
kNN = make_pipeline(RobustScaler(),KNeighborsClassifier(n_neighbors=13) )
score = cross_val_score(kNN, train_data, y_label, cv= k_fold, n_jobs =1 , scoring='accuracy')
print(np.mean(score))
```

    0.8047066167290886



```python
RandomForest = make_pipeline(RobustScaler(),RandomForestClassifier(n_estimators=13) )
score = cross_val_score(RandomForest, train_data, y_label, cv= k_fold, n_jobs =1 , scoring='accuracy')
print(np.mean(score))
```

    0.7979525593008739



```python
Bayes = make_pipeline(RobustScaler(),GaussianNB())
score = cross_val_score(Bayes, train_data, y_label, cv= k_fold, n_jobs =1 ,  scoring='accuracy')
print(np.mean(score))
```

    0.6971161048689138



```python
SV_clf = make_pipeline(RobustScaler(),SVC())
score = cross_val_score(SV_clf, train_data, y_label, cv= k_fold, n_jobs =1 ,  scoring='accuracy')
print(np.mean(score))
```

    0.8338826466916354



```python
#   SVM 모델이 정확도가 제일 좋으니까 83%로,, 이걸로 예측을 하자!
clf =  make_pipeline(RobustScaler(),SVC())
clf.fit(train_data, y_label)
train_prediction = clf.predict(train_data)
test_prediction = clf.predict(test_data)
```


```python
#plot between predicted values and label
error = abs(train_prediction - y_label)
error = pd.Series(error)
error = pd.DataFrame(error.value_counts().values, index=error.value_counts().index.map({0:"True", 1:"False"}), columns=['Count'])
error.plot(kind='bar', figsize=(10,5))
```




    <AxesSubplot:>




    
![svg](/assets/titanic_my_files/titanic_my_51_1.svg)
    



```python
test_prediction=test_prediction.astype(np.int)
```


```python
# 출력하기
submission = pd.DataFrame({"PassengerId": test['PassengerId'],
  "Survived": test_prediction})

submission.to_csv('submission_wook.csv',index=False)
```


```python
submission = pd.read_csv('submission_wook.csv')
submission.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## 끝