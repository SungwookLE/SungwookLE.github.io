---
layout: post
type: research
date: 2021-06-29 10:10
category: Kaggle
title: DataAnalysis sklearn_TypeC
subtitle: Clustering- KMeans 다뤄보자
writer: 100
post-header: true
header-img: 
hash-tag: [Kaggle, Classifier]
use_math: true
---

# sklearn_TypeC
- AUTHOR: SungwookLE  
- DATE: '21.6/29  

- 문제:  
1. 제공된 데이터를 이용하여 유사한 데이터끼리 묶는 군집화 수행  
2. 군집의 대표값을 추출하여 제출 문서에 작성  
3. input_test.csv 파일로 만든 군집화 모델의 출력을 output_test.csv로 저장하시오  


```python
from subprocess import check_output
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
#print(check_output(['ls']).decode('utf8'))
```


```python
input_train = pd.read_csv('input_train.csv', header=None)
input_test = pd.read_csv('input_test.csv', header=None)

print(input_train.shape)
print(input_test.shape)
```

    (10000, 5)
    (2000, 5)


## 1. KMeans CLUSTERING
- n_clustering = 2


```python
from sklearn.cluster import KMeans
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

stscaler = StandardScaler()
stscaler.fit(input_train)

data_train= stscaler.transform(input_train)
data_train= pd.DataFrame(data_train)

data_test = stscaler.transform(input_test)
data_test= pd.DataFrame(data_test)

kM = KMeans(n_clusters=2, algorithm='auto', random_state=0)
kM.fit(data_train)

pred = kM.predict(data_train)


```

## 2. Cluster Representative Value


```python
cluster_centers_ = kM.cluster_centers_
cluster_centers_ = stscaler.inverse_transform(cluster_centers_) #stscaler 도메인으로 transform 했었으니까, 원래 데이터 도메인으로 inverse 해주어야함
print("Cluster Representative Values are {} and {}".format(cluster_centers_[0],cluster_centers_[1]))
```

    Cluster Representative Values are [-0.00696411 -0.78155713  1.11747138  0.00709656 -0.0305657 ] and [-0.02538592  0.59233341 -0.84742607 -0.01145731 -0.02364979]


## 3. VISUALIZATION PLOT
- 학습된 라벨을 기준으로, 데이터가 대표성을 띄고 있는지 본 것 using `PCA`
- 이걸로 보았을 때, 잘 분류되었다는 것은 clustering이 잘되었단 의미


```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(data_train)
X= pca.transform(data_train)
plt.scatter(x=X[:,0], y=X[:,1], c=pred, s=1)

import numpy as np
cumsum=np.cumsum(pca.explained_variance_ratio_)
print('PCA features has {} information'.format(cumsum[-1]))
```

    PCA features has 0.5310049287682159 information



    
![svg](/assets/AI_Compete_TypeC_files/sklearn_prob3_8_1.svg)
    


## 4. Test the model and Save Output File


```python
pred = kM.predict(data_test)
sub = pd.DataFrame({'Label':pred})
```


```python
sub.to_csv('output_test.csv', index=False, header=False)
double_check= pd.read_csv('output_test.csv', header=None)
```
f

```python
double_check.head()
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
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## 끝


