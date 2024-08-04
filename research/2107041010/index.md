---
layout: post
type: research
date: 2021-07-04 10:10
category: Kaggle
title: DataAnalysis Kaggle HeartAttack
subtitle: Classifier- Heart Attack probability high or low?
writer: 100
post-header: true
header-img: 
hash-tag: [Kaggle, Classifier]
use_math: true
---

# KAGGLE: HEART ATTACK PREDICT
AUTHOR: SungwookLE  
DATE: '21.7/4  
DATASET: https://www.kaggle.com/rashikrahmanpritom/heart-attack-analysis-prediction-dataset  

![heart](https://storage.googleapis.com/kaggle-datasets-images/1226038/2046696/2465e7cd117a6954befa50eff39d236f/dataset-cover.jpg?t=2021-03-22-11-33-17)
### About this dataset  
- Age : Age of the patient  
- Sex : Sex of the patient  
- exang: exercise induced angina (1 = yes; 0 = no)  
- ca: number of major vessels (0-3)  
- cp : Chest Pain type chest pain type  
    Value 1: typical angina  
    Value 2: atypical angina  
    Value 3: non-anginal pain  
    Value 4: asymptomatic  
  
- trtbps : resting blood pressure (in mm Hg)
- chol : cholestoral in mg/dl fetched via BMI sensor
- fbs : (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
- rest_ecg : resting electrocardiographic results
    Value 0: normal  
    Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)  
    Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria  
  
- thalach : maximum heart rate achieved  
- target :  
    0= less chance of heart attack  
    1= more chance of heart attack  
  
  
   
### OVERVIEW
1) Data Analysis
- 데이터 차원, 형태 파악하기
- 그래프 그려서 에측변수 ``와 다른 변수와의 상관관계 파악하기

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


```python
import numpy as np
import pandas as pd
from subprocess import check_output

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()
```


```python
print(check_output(["ls","input"]).decode('utf8'))
```

    heart.csv
    o2Saturation.csv
    


## 1) Data Analysis
데이터 차원, 형태 파악하기  
그래프 그려서 에측변수 output 다른 변수와의 상관관계 파악하기  


```python
heart = pd.read_csv("input/heart.csv")
o2Saturation = pd.read_csv("input/o2Saturation.csv")
```


```python
heart.shape, o2Saturation.shape
```




    ((303, 14), (3585, 1))




```python
corr = heart.corr()
f, ax = plt.subplots(figsize=(10,8))
sns.heatmap(corr, vmax=0.8, annot=True)
abs(corr['output']).sort_values(ascending=False)
```




    output      1.000000
    exng        0.436757
    cp          0.433798
    oldpeak     0.430696
    thalachh    0.421741
    caa         0.391724
    slp         0.345877
    thall       0.344029
    sex         0.280937
    age         0.225439
    trtbps      0.144931
    restecg     0.137230
    chol        0.085239
    fbs         0.028046
    Name: output, dtype: float64




    
![svg](/assets/heartattack_files/heartattack_6_1.svg)
    



```python
# ALL DATA TYPES are numerical data
heart.dtypes
```




    age           int64
    sex           int64
    cp            int64
    trtbps        int64
    chol          int64
    fbs           int64
    restecg       int64
    thalachh      int64
    exng          int64
    oldpeak     float64
    slp           int64
    caa           int64
    thall         int64
    output        int64
    dtype: object




```python
heart.isnull().sum()
```




    age         0
    sex         0
    cp          0
    trtbps      0
    chol        0
    fbs         0
    restecg     0
    thalachh    0
    exng        0
    oldpeak     0
    slp         0
    caa         0
    thall       0
    output      0
    dtype: int64




```python
def facet_plot(feature, range_opt=None):
    facet = sns.FacetGrid(heart, hue='output', aspect=4)
    facet.map(sns.kdeplot, feature, shade = True)

    if not range_opt:
        facet.set(xlim=(0, heart[feature].max()))
    else:
        facet.set(xlim=range_opt)
    facet.add_legend()
    plt.title("Output: "+feature)
```


```python
for i in heart.columns:
    if ( i != 'output'):
        facet_plot(i)
```


    
![svg](/assets/heartattack_files/heartattack_10_0.svg)
    



    
![svg](/assets/heartattack_files/heartattack_10_1.svg)
    



    
![svg](/assets/heartattack_files/heartattack_10_2.svg)
    



    
![svg](/assets/heartattack_files/heartattack_10_3.svg)
    



    
![svg](/assets/heartattack_files/heartattack_10_4.svg)
    



    
![svg](/assets/heartattack_files/heartattack_10_5.svg)
    



    
![svg](/assets/heartattack_files/heartattack_10_6.svg)
    



    
![svg](/assets/heartattack_files/heartattack_10_7.svg)
    



    
![svg](/assets/heartattack_files/heartattack_10_8.svg)
    



    
![svg](/assets/heartattack_files/heartattack_10_9.svg)
    



    
![svg](/assets/heartattack_files/heartattack_10_10.svg)
    



    
![svg](/assets/heartattack_files/heartattack_10_11.svg)
    



    
![svg](/assets/heartattack_files/heartattack_10_12.svg)
    



```python
heart['cp'].value_counts()
```




    0    143
    2     87
    1     50
    3     23
    Name: cp, dtype: int64



## 2) Feature Engineering  
### 2-1) categorical + numerical features 분리하기
- using `select_dtypes()`.    
- numerical 데이터 중 month나 year 등의 데이터는 categorical로 분류해주기 `apply(str)`  


```python
heart['sex']=heart['sex'].apply(str)
# 사실 안해줘도 되는데, 그냥 해준거
```


```python
heart.dtypes
```




    age           int64
    sex          object
    cp            int64
    trtbps        int64
    chol          int64
    fbs           int64
    restecg       int64
    thalachh      int64
    exng          int64
    oldpeak     float64
    slp           int64
    caa           int64
    thall         int64
    output        int64
    dtype: object




```python
from sklearn.preprocessing import LabelEncoder

lbl = LabelEncoder()
lbl.fit(heart['sex'])
heart['sex'] =  lbl.transform(heart['sex'].values)
```


```python
heart['sex'].value_counts()
```




    1    207
    0     96
    Name: sex, dtype: int64



### 2-2) 비어있는 missing 데이터 채우기
- numerical: mean, median, mode 를 활용하여 데이터 채우기 .fillna(xxx), mean(), median(), mode()
- categorical: pd.get_dummies() 나 LabelEncoder를 활용해서 missing 데이터도 없애고, one-hot encoding도 완성하기

**비어있는 데이터가 없네요,**

### 2-3) data의 skewness 줄이기
- numerical data 의 skewness 줄이기


```python
#비어있는 데이터 없음
heart.isnull().sum()
```




    age         0
    sex         0
    cp          0
    trtbps      0
    chol        0
    fbs         0
    restecg     0
    thalachh    0
    exng        0
    oldpeak     0
    slp         0
    caa         0
    thall       0
    output      0
    dtype: int64




```python
from scipy import stats
from scipy.stats import norm, skew # for some statistics
```


```python
skewness = heart.apply(lambda x: skew(x.dropna()))
skewness = skewness.sort_values(ascending=False)
skewness_features = skewness[abs(skewness.values)>1].index
print("skewness:")
print(skewness_features)
```

    skewness:
    Index(['fbs', 'caa', 'oldpeak', 'chol'], dtype='object')



```python
sns.distplot(heart['oldpeak'], fit=norm)
(mu,sigma)=norm.fit(heart['oldpeak'])
plt.legend(['Normal dist. mu={:.2f}, std={:.2f}'.format(mu,sigma)], loc='best')
plt.ylabel('Frequency')
plt.title('Before skewness in oldpeak, skew is {:.3f}'.format(skewness['oldpeak']))
plt.show()

heart['oldpeak'] = np.log1p(heart['oldpeak'])
sns.distplot(heart['oldpeak'], fit=norm)
(mu,sigma)=norm.fit(heart['oldpeak'])
plt.legend(['Normal dist. mu={:.2f}, std={:.2f}'.format(mu,sigma)], loc='best')
plt.ylabel('Frequency')
plt.title('After skewness in oldpeak, skew is  {:.3f}'.format(skew(heart['oldpeak'])))
plt.show()

sns.distplot(heart['chol'], fit=norm)
(mu,sigma)=norm.fit(heart['chol'])
plt.legend(['Normal dist. mu={:.2f}, std={:.2f}'.format(mu,sigma)], loc='best')
plt.ylabel('Frequency')
plt.title('Before skewness in chol, skew is {:.3f}'.format(skewness['chol']))
plt.show()

heart['chol'] = np.log1p(heart['chol'])
sns.distplot(heart['chol'], fit=norm)
(mu,sigma)=norm.fit(heart['chol'])
plt.legend(['Normal dist. mu={:.2f}, std={:.2f}'.format(mu,sigma)], loc='best')
plt.ylabel('Frequency')
plt.title('After skewness in chol, skew is {:.3f}'.format(skew(heart['chol'])))
plt.show()
```


    
![svg](/assets/heartattack_files/heartattack_21_0.svg)
    



    
![svg](/assets/heartattack_files/heartattack_21_1.svg)
    



    
![svg](/assets/heartattack_files/heartattack_21_2.svg)
    



    
![svg](/assets/heartattack_files/heartattack_21_3.svg)
    



```python
#QQ plots
fig = plt.figure()
res = stats.probplot(heart['chol'], plot=plt)
plt.show()
```


    
![svg](/assets/heartattack_files/heartattack_22_0.svg)
    



```python
skewness = heart.apply(lambda x: skew(x.dropna()))
skewness = skewness.sort_values(ascending=False)
skewness_features = skewness[abs(skewness.values)>1].index
print("skewness:")
print(skewness_features)
```

    skewness:
    Index(['fbs', 'caa'], dtype='object')


## 3) Modeling
- CrossValidation using `cross_val_score, KFold. train_test_split`.          
- Regressor : `LinearRegression, RidgeCV, LassoCV, ElasticNetCV` 
- Classifier  : `KNN, RandomForest, MLPClassifier...`
- Techniques: `StandardScaler, RobustScaler`.
- Easy modeling: `make_pipeline`


```python
label = heart['output']
heart.drop('output',axis=1, inplace=True)
heart.head()
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
      <th>age</th>
      <th>sex</th>
      <th>cp</th>
      <th>trtbps</th>
      <th>chol</th>
      <th>fbs</th>
      <th>restecg</th>
      <th>thalachh</th>
      <th>exng</th>
      <th>oldpeak</th>
      <th>slp</th>
      <th>caa</th>
      <th>thall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>63</td>
      <td>1</td>
      <td>3</td>
      <td>145</td>
      <td>5.455321</td>
      <td>1</td>
      <td>0</td>
      <td>150</td>
      <td>0</td>
      <td>1.193922</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>37</td>
      <td>1</td>
      <td>2</td>
      <td>130</td>
      <td>5.525453</td>
      <td>0</td>
      <td>1</td>
      <td>187</td>
      <td>0</td>
      <td>1.504077</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>41</td>
      <td>0</td>
      <td>1</td>
      <td>130</td>
      <td>5.323010</td>
      <td>0</td>
      <td>0</td>
      <td>172</td>
      <td>0</td>
      <td>0.875469</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>56</td>
      <td>1</td>
      <td>1</td>
      <td>120</td>
      <td>5.468060</td>
      <td>0</td>
      <td>1</td>
      <td>178</td>
      <td>0</td>
      <td>0.587787</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>57</td>
      <td>0</td>
      <td>0</td>
      <td>120</td>
      <td>5.872118</td>
      <td>0</td>
      <td>1</td>
      <td>163</td>
      <td>1</td>
      <td>0.470004</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

k_fold = KFold(n_splits = 10, shuffle = True, random_state = 0)
```


```python
RandomForest = make_pipeline(StandardScaler(),RandomForestClassifier(n_estimators=50) )
score = cross_val_score(RandomForest, heart, label, cv= k_fold, n_jobs =1 , scoring='accuracy')
print('RandomForest CrossValidation Score is {:.5f}'.format(np.mean(score)))
```

    RandomForest CrossValidation Score is 0.82215



```python
MLP = make_pipeline(StandardScaler(), MLPClassifier(learning_rate='adaptive'))
score = cross_val_score(MLP, heart, label, cv= k_fold, n_jobs =1 , scoring='accuracy')
print('MLP CrossValidation Score is {:.5f}'.format(np.mean(score)))
```

    MLP CrossValidation Score is 0.84172


- Best Model is SVM as 83.495%


```python
SVM = make_pipeline(StandardScaler(), SVC())
score = cross_val_score(SVM, heart, label, cv= k_fold, n_jobs =1 , scoring='accuracy')
print('SVM CrossValidation Score is {:.5f}'.format(np.mean(score)))
```

    SVM CrossValidation Score is 0.83495



```python
KNN = make_pipeline(StandardScaler(), KNeighborsClassifier())
score = cross_val_score(KNN, heart, label, cv= k_fold, n_jobs =1 , scoring='accuracy')
print('KNN CrossValidation Score is {:.5f}'.format(np.mean(score)))
```

    KNN CrossValidation Score is 0.83194



```python
sts = StandardScaler()
sts.fit(heart)
feed = sts.transform(heart)
feed = pd.DataFrame(feed, columns= heart.columns)

#skewness= feed.apply(lambda x: skew(x.dropna()))
#print(skewness)
#from sklearn.cluster import KMeans
#KMM = KMeans(n_clusters=2)
#KMM.fit(feed)

SVM.fit(feed, label)
pred = SVM.predict(feed)
pred = pd.DataFrame(pred, columns=['output'])

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(feed)

X = pca.transform(feed)
X= pd.DataFrame(X)
X=X.rename(columns={0:'one', 1:'two'})
X = pd.concat([X, pred], axis=1)
```


```python
colors = ['red','blue']
labels=[0,1]
legends = ['less chance of heart atk.', 'more chance of heart atk.']

for la, color, leg in zip(labels, colors, legends)W:
    plt.scatter(x=X.loc[X['output']==la]['one'], y = X.loc[X['output']==la]['two'], c =color, s=3, label=leg)
plt.legend()

```




    <matplotlib.legend.Legend at 0x7f5afa0d7b38>




    
![svg](/assets/heartattack_files/heartattack_33_1.svg)
    


## 끝
- 데이터 프로세스 & 기계학습 순서를 따라가면, 어느정도의 predict 성능은 나오는데, 83% 정도에서 더 끌어올리기 위해선, **feature engineering**을 신경써서 해주어야 한다.
