---
layout: post
type: research
date: 2021-06-28 10:10
category: Kaggle
title: DataAnalysis sklearn_TypeB
subtitle: PCA- Feature Dimenssion Reduction 다뤄보자
writer: 100
post-header: true
header-img: 
hash-tag: [Kaggle, PCA]
use_math: true
---

# sklearn_TypeB
- AUTHOR: SungwookLE   
- DATE: '21.6/28   

문제:   
1. 5차원으로 데이터의 차원 축소   
2. 축소를 통해 정리된 인자들의 특성을 살펴라    

**OVEVIEW**  
1. Data Load and View  
2. Feature Dimenstion Reduce  
3. Feature Character    

## 1. Data Load and View


```python
from subprocess import check_output
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()
from sklearn.preprocessing import StandardScaler

print(check_output(["ls","input"]).decode('utf8'))
```

    AI경진대회 예선(B형)_data-set.zip
    input.csv
    output.csv
    



```python
input = pd.read_csv('input/input.csv', header=None)
output = pd.read_csv('input/output.csv', header=None)
output = output.rename(columns={0:'Label'})
```


```python
input.values
```




    array([[  0.83035958,  -0.33025241,  -0.23054277, ...,  -1.02979077,
             -4.27514811,  -0.59929727],
           [ -0.04399859,   0.22065793,   1.60051901, ...,  -1.10753423,
            -20.25542908,  -0.56636377],
           [  0.62671752,   2.10042501,  -0.96579802, ...,  -1.03976259,
            -10.22693074,  -1.05338458],
           ...,
           [ -0.71817134,   0.26945901,   0.53723753, ...,   0.44234589,
            -13.406614  ,   0.85427125],
           [ -0.3884856 ,  -0.20375512,   1.40039956, ...,  -1.08230872,
             40.66522873,  -1.58154278],
           [ -0.09540666,   1.47321441,   1.05998807, ...,  -1.11836725,
             27.74371615,  -1.51622948]])




```python
# Standard Scaling: Data Normalization -> set mean 0 , and std 1.
stscaler= StandardScaler()
stscaler.fit(input)
input_ = stscaler.transform(input)

y = output
data = pd.DataFrame(input_)
data.head()
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
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.818106</td>
      <td>-0.325165</td>
      <td>-0.246214</td>
      <td>-0.012640</td>
      <td>-0.736274</td>
      <td>1.205771</td>
      <td>-1.074941</td>
      <td>0.554851</td>
      <td>1.166372</td>
      <td>0.445609</td>
      <td>1.065898</td>
      <td>0.114682</td>
      <td>-0.506004</td>
      <td>0.006854</td>
      <td>1.423487</td>
      <td>-0.674604</td>
      <td>0.299847</td>
      <td>-1.193585</td>
      <td>-0.215506</td>
      <td>-0.523844</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.051535</td>
      <td>0.224741</td>
      <td>1.586533</td>
      <td>-0.806663</td>
      <td>-1.905261</td>
      <td>-1.428647</td>
      <td>-1.138761</td>
      <td>-0.189653</td>
      <td>1.009079</td>
      <td>1.932479</td>
      <td>0.060460</td>
      <td>1.766632</td>
      <td>1.395304</td>
      <td>-0.959874</td>
      <td>0.196900</td>
      <td>-0.570431</td>
      <td>0.004655</td>
      <td>-1.283734</td>
      <td>-1.005080</td>
      <td>-0.494709</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.615563</td>
      <td>2.101084</td>
      <td>-0.982146</td>
      <td>-0.264738</td>
      <td>1.043126</td>
      <td>2.171876</td>
      <td>-1.185627</td>
      <td>0.103988</td>
      <td>0.612144</td>
      <td>-0.328839</td>
      <td>0.849970</td>
      <td>-0.459181</td>
      <td>-1.007995</td>
      <td>1.222779</td>
      <td>-0.832886</td>
      <td>0.692303</td>
      <td>-1.271343</td>
      <td>-1.205148</td>
      <td>-0.509579</td>
      <td>-0.925548</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.607127</td>
      <td>-2.041983</td>
      <td>0.181954</td>
      <td>-0.300708</td>
      <td>-1.254648</td>
      <td>-2.385617</td>
      <td>-0.814498</td>
      <td>-0.312310</td>
      <td>1.381110</td>
      <td>-0.885105</td>
      <td>-0.578846</td>
      <td>-0.599110</td>
      <td>0.880172</td>
      <td>-1.801167</td>
      <td>0.113969</td>
      <td>0.472810</td>
      <td>-0.432572</td>
      <td>-0.787749</td>
      <td>1.030029</td>
      <td>-0.761267</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.416870</td>
      <td>0.361684</td>
      <td>0.424445</td>
      <td>-1.209425</td>
      <td>0.349353</td>
      <td>-0.397125</td>
      <td>-0.460240</td>
      <td>0.267333</td>
      <td>-1.291405</td>
      <td>1.225944</td>
      <td>1.432778</td>
      <td>0.157923</td>
      <td>-0.985794</td>
      <td>0.625482</td>
      <td>-0.446556</td>
      <td>-2.052565</td>
      <td>-0.418550</td>
      <td>-0.468195</td>
      <td>1.015300</td>
      <td>-0.358104</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.describe()
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
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.000000e+04</td>
      <td>1.000000e+04</td>
      <td>1.000000e+04</td>
      <td>1.000000e+04</td>
      <td>1.000000e+04</td>
      <td>1.000000e+04</td>
      <td>1.000000e+04</td>
      <td>1.000000e+04</td>
      <td>1.000000e+04</td>
      <td>1.000000e+04</td>
      <td>1.000000e+04</td>
      <td>1.000000e+04</td>
      <td>1.000000e+04</td>
      <td>1.000000e+04</td>
      <td>1.000000e+04</td>
      <td>1.000000e+04</td>
      <td>1.000000e+04</td>
      <td>1.000000e+04</td>
      <td>1.000000e+04</td>
      <td>1.000000e+04</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-3.087669e-17</td>
      <td>-1.141309e-17</td>
      <td>9.348078e-18</td>
      <td>-5.060535e-17</td>
      <td>3.215206e-17</td>
      <td>3.910205e-17</td>
      <td>-1.594280e-17</td>
      <td>-4.818368e-18</td>
      <td>6.561418e-18</td>
      <td>2.553513e-17</td>
      <td>7.327472e-19</td>
      <td>-2.753353e-18</td>
      <td>-3.052072e-18</td>
      <td>-3.863576e-18</td>
      <td>-3.530856e-17</td>
      <td>-3.313044e-17</td>
      <td>3.850253e-17</td>
      <td>1.809664e-17</td>
      <td>-4.207745e-17</td>
      <td>4.478640e-17</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.000050e+00</td>
      <td>1.000050e+00</td>
      <td>1.000050e+00</td>
      <td>1.000050e+00</td>
      <td>1.000050e+00</td>
      <td>1.000050e+00</td>
      <td>1.000050e+00</td>
      <td>1.000050e+00</td>
      <td>1.000050e+00</td>
      <td>1.000050e+00</td>
      <td>1.000050e+00</td>
      <td>1.000050e+00</td>
      <td>1.000050e+00</td>
      <td>1.000050e+00</td>
      <td>1.000050e+00</td>
      <td>1.000050e+00</td>
      <td>1.000050e+00</td>
      <td>1.000050e+00</td>
      <td>1.000050e+00</td>
      <td>1.000050e+00</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-3.688757e+00</td>
      <td>-3.859865e+00</td>
      <td>-4.714368e+00</td>
      <td>-3.885379e+00</td>
      <td>-3.629213e+00</td>
      <td>-3.770522e+00</td>
      <td>-1.870203e+00</td>
      <td>-3.621609e+00</td>
      <td>-3.600433e+00</td>
      <td>-3.997427e+00</td>
      <td>-3.868648e+00</td>
      <td>-3.610129e+00</td>
      <td>-3.593709e+00</td>
      <td>-3.621035e+00</td>
      <td>-3.912132e+00</td>
      <td>-3.924080e+00</td>
      <td>-3.670515e+00</td>
      <td>-1.952642e+00</td>
      <td>-3.565266e+00</td>
      <td>-2.146557e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-6.701758e-01</td>
      <td>-6.670909e-01</td>
      <td>-6.835581e-01</td>
      <td>-6.800196e-01</td>
      <td>-6.682490e-01</td>
      <td>-6.670017e-01</td>
      <td>-9.264151e-01</td>
      <td>-6.859074e-01</td>
      <td>-6.698788e-01</td>
      <td>-6.765303e-01</td>
      <td>-6.758763e-01</td>
      <td>-6.712561e-01</td>
      <td>-6.837196e-01</td>
      <td>-6.817490e-01</td>
      <td>-6.791238e-01</td>
      <td>-6.738840e-01</td>
      <td>-6.684718e-01</td>
      <td>-9.895199e-01</td>
      <td>-6.780318e-01</td>
      <td>-8.897937e-01</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.088883e-03</td>
      <td>4.061022e-03</td>
      <td>6.491199e-04</td>
      <td>3.440946e-03</td>
      <td>-3.559668e-03</td>
      <td>6.739244e-03</td>
      <td>2.121647e-03</td>
      <td>-1.018180e-02</td>
      <td>8.129254e-04</td>
      <td>1.343290e-02</td>
      <td>-6.378701e-03</td>
      <td>-1.497902e-03</td>
      <td>1.165557e-03</td>
      <td>9.576464e-03</td>
      <td>8.139716e-03</td>
      <td>-6.609107e-03</td>
      <td>-8.836017e-03</td>
      <td>5.153569e-02</td>
      <td>-2.638754e-03</td>
      <td>-1.884091e-01</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.776271e-01</td>
      <td>6.633211e-01</td>
      <td>6.685248e-01</td>
      <td>6.733688e-01</td>
      <td>6.826127e-01</td>
      <td>6.669742e-01</td>
      <td>6.198403e-01</td>
      <td>6.767314e-01</td>
      <td>6.737767e-01</td>
      <td>6.555822e-01</td>
      <td>6.713280e-01</td>
      <td>6.741036e-01</td>
      <td>6.808373e-01</td>
      <td>6.771723e-01</td>
      <td>6.740690e-01</td>
      <td>6.795435e-01</td>
      <td>6.981673e-01</td>
      <td>5.261333e-01</td>
      <td>6.744299e-01</td>
      <td>8.889074e-01</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.786736e+00</td>
      <td>3.700328e+00</td>
      <td>3.819858e+00</td>
      <td>3.731669e+00</td>
      <td>3.686054e+00</td>
      <td>3.725068e+00</td>
      <td>3.512268e+00</td>
      <td>3.896716e+00</td>
      <td>3.842502e+00</td>
      <td>4.236235e+00</td>
      <td>3.556807e+00</td>
      <td>3.783354e+00</td>
      <td>3.717151e+00</td>
      <td>3.382842e+00</td>
      <td>3.849388e+00</td>
      <td>3.861931e+00</td>
      <td>3.529508e+00</td>
      <td>3.719336e+00</td>
      <td>3.887281e+00</td>
      <td>3.033482e+00</td>
    </tr>
  </tbody>
</table>
</div>



## 2. Feature Dimenstion Reduce
- using `PCA`: one of represenative method known as Linear Dimension Reduction. PCA(Principal Component Analysis)


```python
from sklearn.decomposition import PCA

pca = PCA(n_components=5)
reduced_dim = pca.fit_transform(data)
reduced_dim = pd.DataFrame(reduced_dim, columns=['principal_comp1','principal_comp2','principal_comp3','principal_comp4','principal_comp5'])
reduced_dim
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
      <th>principal_comp1</th>
      <th>principal_comp2</th>
      <th>principal_comp3</th>
      <th>principal_comp4</th>
      <th>principal_comp5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.618868</td>
      <td>0.604217</td>
      <td>0.781736</td>
      <td>-0.547131</td>
      <td>-0.846699</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.720829</td>
      <td>-0.031994</td>
      <td>0.435634</td>
      <td>-2.228988</td>
      <td>0.958939</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.926017</td>
      <td>-1.115552</td>
      <td>0.746879</td>
      <td>2.104426</td>
      <td>-1.363396</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.350258</td>
      <td>0.556883</td>
      <td>-0.693035</td>
      <td>-2.326694</td>
      <td>1.298485</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.746567</td>
      <td>0.096190</td>
      <td>-0.198689</td>
      <td>1.189678</td>
      <td>-1.028490</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9995</th>
      <td>3.045021</td>
      <td>0.325163</td>
      <td>-0.428845</td>
      <td>1.365473</td>
      <td>-1.105546</td>
    </tr>
    <tr>
      <th>9996</th>
      <td>-0.575423</td>
      <td>0.545010</td>
      <td>1.186840</td>
      <td>0.204053</td>
      <td>-0.563458</td>
    </tr>
    <tr>
      <th>9997</th>
      <td>1.088324</td>
      <td>-0.578450</td>
      <td>0.264246</td>
      <td>-0.760159</td>
      <td>-0.259600</td>
    </tr>
    <tr>
      <th>9998</th>
      <td>-2.304876</td>
      <td>-0.420677</td>
      <td>-1.557204</td>
      <td>1.736730</td>
      <td>1.008181</td>
    </tr>
    <tr>
      <th>9999</th>
      <td>-2.319175</td>
      <td>0.200419</td>
      <td>0.456647</td>
      <td>0.820593</td>
      <td>-0.812472</td>
    </tr>
  </tbody>
</table>
<p>10000 rows × 5 columns</p>
</div>



## 3. Feature Character
1) Test Classifier using RandomForestClassifier   
2) Charateristic: explained variance ratio summation is 34.703%, i.e 34.7% information of original data was remained


```python
from sklearn.ensemble import RandomForestClassifier
# 1) Test Classifier using RandomForestClassifier 
clf=RandomForestClassifier(n_estimators=90)
clf.fit(reduced_dim,y)

score = clf.score(reduced_dim, y)
print("RandomForestClassifier Score is {} %".format(score*100))
print("Dimension Reduction was executed well!")

```

    RandomForestClassifier Score is 100.0 %
    Dimension Reduction was executed well!



```python
import numpy as np
# 2) Charateristic
print('Eigen_value :', pca.explained_variance_)
#Explained Variance Ratio는 주성분 벡터가 이루는 축에 투영(projection)한 결과의 분산의 비율을 말하며, 각 eigenvalue의 비율과 같은 의미
print('Explained variance ratio :', pca.explained_variance_ratio_) 
print('Information Reduction Percet is {}%'.format(np.sum(pca.explained_variance_ratio_)*100))

df = pd.DataFrame(pca.explained_variance_ratio_, index=['principal_comp1','principal_comp2','principal_comp3','principal_comp4','principal_comp5'], columns=['explained_variance_ratio_'])
df.plot.pie(y='explained_variance_ratio_', figsize=(7,7), legend=False)
```

    Eigen_value : [2.76572688 1.05425708 1.04845199 1.03346204 1.02878758]
    Explained variance ratio : [0.13827252 0.05270758 0.05241736 0.05166793 0.05143424]
    Information Reduction Percet is 34.64996247376067%





    <AxesSubplot:ylabel='explained_variance_ratio_'>




    
![svg](/assets/AI_Compete_TypeB_files/AI_Compete_TypeB_10_2.svg)
    



```python
plt.figure()
colors=['navy','red']

aug = pd.concat([reduced_dim, y],axis=1)

fig=plt.figure(figsize=(30,10))
ax1=fig.add_subplot(1,4,1)
ax2=fig.add_subplot(1,4,2)
ax3=fig.add_subplot(1,4,3)
ax4=fig.add_subplot(1,4,4)

for color, label in zip(colors, [0, 1]):
    ax1.scatter(x=aug.loc[aug['Label'] == label]['principal_comp1'], y=aug.loc[aug['Label'] == label]['principal_comp2'], c=color,  s=1, alpha=0.9, label=str(label))
    ax1.set_xlabel('principal_comp1')
    ax1.set_ylabel('principal_comp2')

for color, label in zip(colors, [0, 1]):
    ax2.scatter(x=aug.loc[aug['Label'] == label]['principal_comp1'], y=aug.loc[aug['Label'] == label]['principal_comp3'], c=color,  s=1, alpha=0.9)
    ax2.set_xlabel('principal_comp1')
    ax2.set_ylabel('principal_comp3')

for color, label in zip(colors, [0, 1]):
    ax3.scatter(x=aug.loc[aug['Label'] == label]['principal_comp1'], y=aug.loc[aug['Label'] == label]['principal_comp4'], c=color,  s=1, alpha=0.9)
    ax3.set_xlabel('principal_comp1')
    ax3.set_ylabel('principal_comp4')

for color, label in zip(colors, [0, 1]):
    ax4.scatter(x=aug.loc[aug['Label'] == label]['principal_comp1'], y=aug.loc[aug['Label'] == label]['principal_comp5'], c=color,  s=1, alpha=0.9)
    ax4.set_xlabel('principal_comp1')
    ax4.set_ylabel('principal_comp5')

```


    <Figure size 432x288 with 0 Axes>



    
![svg](/assets/AI_Compete_TypeB_files/AI_Compete_TypeB_11_1.svg)
    



```python
corrmat = aug.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corrmat, annot=True)
```




    <AxesSubplot:>




    
![svg](/assets/AI_Compete_TypeB_files/AI_Compete_TypeB_12_1.svg)
    


## 끝


```python

```
