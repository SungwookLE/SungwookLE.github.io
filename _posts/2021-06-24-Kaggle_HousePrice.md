---
title:  "DataAnalysis: Kaggle HousePrice"
excerpt: "Kaggle: HousePrice 문제, Titanic은 logistic이엇다면 이번 문제는 regression"

categories:
  - research
tags:
  - research

toc: true
toc_sticky: true
 
date: 2021-06-24
---

### Data Analysis: Kaggle HousePrice
> AUTHOR: SungwookLE  
> DATE: '21.6/24

아래는, 간단하게, 파이썬 주피터노트북 파일을 마크다운 변환한 것  

# HOUSE PRICES PREDICTION
- DATE: '21.6/21  
- AUTHOR: SungwookLE(joker1251@naver.com)  
- REFERENCE: https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard

![image](https://storage.googleapis.com/kaggle-competitions/kaggle/5407/media/housesbanner.png)

Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.

With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.

## 1. Goal
It is your job to predict the sales price for each house. For each Id in the test set, you must predict the value of the SalePrice variable. 

## 2. Metric
Submissions are evaluated on Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted value and the logarithm of the observed sales price. (Taking logs means that errors in predicting expensive houses and cheap houses will affect the result equally.)

Submission File Format
The file should contain a header and have the following format:
```
Id,SalePrice
1461,169000.1
1462,187724.1233
1463,175221
etc.
```

## 3. File descriptions
- train.csv - the training set
- test.csv - the test set
- data_description.txt - full description of each column, originally prepared by Dean De Cock but lightly edited to match the column names used here
- sample_submission.csv - a benchmark submission from a linear regression on year and month of sale, lot square footage, and number of bedrooms

## 4. Data fields
Here's a brief version of what you'll find in the data description file.  

|**Key**|**Value**|
|:---|:---|
|SalePrice|the property's sale price in dollars. This is the **target** variable that you're trying to predict.|
|MSSubClass|The building class|
|MSZoning|The general zoning classification|
|LotFrontage|Linear feet of street connected to property|
|LotArea|Lot size in square feet|
|Street|Type of road access|
|Alley|Type of alley access|
|LotShape|General shape of property|
|LandContour|Flatness of the property|
|Utilities|Type of utilities available|
|LotConfig|Lot configuration|
|LandSlope|Slope of property|
|Neighborhood|Physical locations within Ames city limits|
|Condition1|Proximity to main road or railroad|
|Condition2|Proximity to main road or railroad (if a second is present)|
|BldgType|Type of dwelling|
|HouseStyle|Style of dwelling|
|OverallQual|Overall material and finish quality|
|OverallCond|Overall condition rating|
|YearBuilt|Original construction date|
|YearRemodAdd|Remodel date|
|RoofStyle|Type of roof|
|RoofMatl|Roof material|
|Exterior1st|Exterior covering on house|
|Exterior2nd|Exterior covering on house (if more than one material)|
|MasVnrType|Masonry veneer type|
|MasVnrArea|Masonry veneer area in square feet|
|ExterQual|Exterior material quality|
|ExterCond|Present condition of the material on the exterior|
|Foundation|Type of foundation|
|BsmtQual|Height of the basement|
|BsmtCond|General condition of the basement|
|BsmtExposure|Walkout or garden level basement walls|
|BsmtFinType1|Quality of basement finished area|
|BsmtFinSF1|Type 1 finished square feet|
|BsmtFinType2|Quality of second finished area (if present)|
|BsmtFinSF2|Type 2 finished square feet|
|BsmtUnfSF|Unfinished square feet of basement area|
|TotalBsmtSF|Total square feet of basement area|
|Heating|Type of heating|
|HeatingQC|Heating quality and condition|
|CentralAir|Central air conditioning|
|Electrical|Electrical system|
|1stFlrSF|First Floor square feet|
|2ndFlrSF|Second floor square feet|
|LowQualFinSF|Low quality finished square feet (all floors)|
|GrLivArea|Above grade (ground) living area square feet|
|BsmtFullBath|Basement full bathrooms|
|BsmtHalfBath|Basement half bathrooms|
|FullBath|Full bathrooms above grade|
|HalfBath|Half baths above grade|
|Bedroom|Number of bedrooms above basement level|
|Kitchen|Number of kitchens|
|KitchenQual|Kitchen quality|
|TotRmsAbvGrd|Total rooms above grade (does not include bathrooms)|
|Functional|Home functionality rating|
|Fireplaces|Number of fireplaces|
|FireplaceQu|Fireplace quality|
|GarageType|Garage location|
|GarageYrBlt|Year garage was built|
|GarageFinish|Interior finish of the garage|
|GarageCars|Size of garage in car capacity|
|GarageArea|Size of garage in square feet|
|GarageQual|Garage quality|
|GarageCond|Garage condition|
|PavedDrive|Paved driveway|
|WoodDeckSF|Wood deck area in square feet|
|OpenPorchSF|Open porch area in square feet|
|EnclosedPorch|Enclosed porch area in square feet|
|3SsnPorch|Three season porch area in square feet|
|ScreenPorch|Screen porch area in square feet|
|PoolArea|Pool area in square feet|
|PoolQC|Pool quality|
|Fence|Fence quality|
|MiscFeature|Miscellaneous feature not covered in other categories|
|MiscVal|$Value of miscellaneous feature|
|MoSold|Month Sold|
|YrSold|Year Sold|
|SaleType|Type of sale|
|SaleCondition|Condition of sale|


```python
import pandas as pd
```


```python
import numpy as np
```


```python
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set() # set default in matplotlib
# 여러가지 Seaborn 그래프: https://dining-developer.tistory.com/30
```


```python
train_original = pd.read_csv('input/train.csv')
test_original = pd.read_csv('input/test.csv')

train = train_original.copy()
test = test_original.copy()
```

## 5. 데이터 살펴보기
- DataFrame의 Columns, index 정보, Shape 보기
- `head()`, `tail()`로 직접 몇개 출력해서 보기
- `describe()`, `info()`, `isnull().sum()`, `value_counts()` 등으로 데이터 살펴보기


```python
print(train.shape)
train.head()
```

    (1460, 81)





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
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>208500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>181500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>223500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>140000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>250000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 81 columns</p>
</div>




```python
print(test.shape)
test.head()
```

    (1459, 80)





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
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1461</td>
      <td>20</td>
      <td>RH</td>
      <td>80.0</td>
      <td>11622</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>120</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1462</td>
      <td>20</td>
      <td>RL</td>
      <td>81.0</td>
      <td>14267</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Gar2</td>
      <td>12500</td>
      <td>6</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1463</td>
      <td>60</td>
      <td>RL</td>
      <td>74.0</td>
      <td>13830</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>3</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1464</td>
      <td>60</td>
      <td>RL</td>
      <td>78.0</td>
      <td>9978</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1465</td>
      <td>120</td>
      <td>RL</td>
      <td>43.0</td>
      <td>5005</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>HLS</td>
      <td>AllPub</td>
      <td>...</td>
      <td>144</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>1</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 80 columns</p>
</div>




```python
# DRAWING DATA WITH INT DATA FOR ANALYSIS
# 숫자 데이터들 중 SalePrice와의 상관관계를 살펴보기 위해 다양한 그림을 그려보자

train_draw = train_original.copy()
def drop_axis(df, feature):
    df.drop(feature, axis=1, inplace=True)

for feature in train_draw.columns:
    if (train.dtypes[feature]=='object'):
        drop_axis(train_draw, feature)

for data in train_draw.isnull().sum().items():
    if(data[1] > 0):
        drop_axis(train_draw, data[0])
```


```python
sn1=sns.pairplot(data=train_draw,\
             x_vars=np.concatenate([ ['SalePrice'] ,train_draw.columns.values[0:5]]),\
             y_vars=['SalePrice'])

sn2=sns.pairplot(data=train_draw,\
             x_vars=np.concatenate([ ['SalePrice'] ,train_draw.columns.values[5:10]]),\
             y_vars=['SalePrice'])
sn3=sns.pairplot(data=train_draw,\
             x_vars=np.concatenate([ ['SalePrice'] ,train_draw.columns.values[10:15]]),\
             y_vars=['SalePrice'])
sn4=sns.pairplot(data=train_draw,\
             x_vars=np.concatenate([ ['SalePrice'] ,train_draw.columns.values[15:20]]),\
             y_vars=['SalePrice'])

sn5=sns.pairplot(data=train_draw,\
             x_vars=np.concatenate([ ['SalePrice'] ,train_draw.columns.values[20:25]]),\
             y_vars=['SalePrice'])

sn6=sns.pairplot(data=train_draw,\
             x_vars=np.concatenate([ ['SalePrice'] ,train_draw.columns.values[25:30]]),\
             y_vars=['SalePrice'])

sn7=sns.pairplot(data=train_draw,\
             x_vars=np.concatenate([ ['SalePrice'] ,train_draw.columns.values[30:]]),\
             y_vars=['SalePrice'])

plt.show()
```


    
![svg](/assets/house_files/house_9_0.svg)
    



    
![svg](/assets/house_files/house_9_1.svg)
    



    
![svg](/assets/house_files/house_9_2.svg)
    



    
![svg](/assets/house_files/house_9_3.svg)
    



    
![svg](/assets/house_files/house_9_4.svg)
    



    
![svg](/assets/house_files/house_9_5.svg)
    



    
![svg](/assets/house_files/house_9_6.svg)
    



```python
# 숫자로 이루어진 데이터들의 수치 정보 확인
train.describe()
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
      <th>Id</th>
      <th>MSSubClass</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>MasVnrArea</th>
      <th>BsmtFinSF1</th>
      <th>...</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1201.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1452.000000</td>
      <td>1460.000000</td>
      <td>...</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>730.500000</td>
      <td>56.897260</td>
      <td>70.049958</td>
      <td>10516.828082</td>
      <td>6.099315</td>
      <td>5.575342</td>
      <td>1971.267808</td>
      <td>1984.865753</td>
      <td>103.685262</td>
      <td>443.639726</td>
      <td>...</td>
      <td>94.244521</td>
      <td>46.660274</td>
      <td>21.954110</td>
      <td>3.409589</td>
      <td>15.060959</td>
      <td>2.758904</td>
      <td>43.489041</td>
      <td>6.321918</td>
      <td>2007.815753</td>
      <td>180921.195890</td>
    </tr>
    <tr>
      <th>std</th>
      <td>421.610009</td>
      <td>42.300571</td>
      <td>24.284752</td>
      <td>9981.264932</td>
      <td>1.382997</td>
      <td>1.112799</td>
      <td>30.202904</td>
      <td>20.645407</td>
      <td>181.066207</td>
      <td>456.098091</td>
      <td>...</td>
      <td>125.338794</td>
      <td>66.256028</td>
      <td>61.119149</td>
      <td>29.317331</td>
      <td>55.757415</td>
      <td>40.177307</td>
      <td>496.123024</td>
      <td>2.703626</td>
      <td>1.328095</td>
      <td>79442.502883</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>20.000000</td>
      <td>21.000000</td>
      <td>1300.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1872.000000</td>
      <td>1950.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2006.000000</td>
      <td>34900.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>365.750000</td>
      <td>20.000000</td>
      <td>59.000000</td>
      <td>7553.500000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>1954.000000</td>
      <td>1967.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>2007.000000</td>
      <td>129975.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>730.500000</td>
      <td>50.000000</td>
      <td>69.000000</td>
      <td>9478.500000</td>
      <td>6.000000</td>
      <td>5.000000</td>
      <td>1973.000000</td>
      <td>1994.000000</td>
      <td>0.000000</td>
      <td>383.500000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>25.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>2008.000000</td>
      <td>163000.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1095.250000</td>
      <td>70.000000</td>
      <td>80.000000</td>
      <td>11601.500000</td>
      <td>7.000000</td>
      <td>6.000000</td>
      <td>2000.000000</td>
      <td>2004.000000</td>
      <td>166.000000</td>
      <td>712.250000</td>
      <td>...</td>
      <td>168.000000</td>
      <td>68.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>8.000000</td>
      <td>2009.000000</td>
      <td>214000.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1460.000000</td>
      <td>190.000000</td>
      <td>313.000000</td>
      <td>215245.000000</td>
      <td>10.000000</td>
      <td>9.000000</td>
      <td>2010.000000</td>
      <td>2010.000000</td>
      <td>1600.000000</td>
      <td>5644.000000</td>
      <td>...</td>
      <td>857.000000</td>
      <td>547.000000</td>
      <td>552.000000</td>
      <td>508.000000</td>
      <td>480.000000</td>
      <td>738.000000</td>
      <td>15500.000000</td>
      <td>12.000000</td>
      <td>2010.000000</td>
      <td>755000.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 38 columns</p>
</div>




```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1460 entries, 0 to 1459
    Data columns (total 81 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   Id             1460 non-null   int64  
     1   MSSubClass     1460 non-null   int64  
     2   MSZoning       1460 non-null   object 
     3   LotFrontage    1201 non-null   float64
     4   LotArea        1460 non-null   int64  
     5   Street         1460 non-null   object 
     6   Alley          91 non-null     object 
     7   LotShape       1460 non-null   object 
     8   LandContour    1460 non-null   object 
     9   Utilities      1460 non-null   object 
     10  LotConfig      1460 non-null   object 
     11  LandSlope      1460 non-null   object 
     12  Neighborhood   1460 non-null   object 
     13  Condition1     1460 non-null   object 
     14  Condition2     1460 non-null   object 
     15  BldgType       1460 non-null   object 
     16  HouseStyle     1460 non-null   object 
     17  OverallQual    1460 non-null   int64  
     18  OverallCond    1460 non-null   int64  
     19  YearBuilt      1460 non-null   int64  
     20  YearRemodAdd   1460 non-null   int64  
     21  RoofStyle      1460 non-null   object 
     22  RoofMatl       1460 non-null   object 
     23  Exterior1st    1460 non-null   object 
     24  Exterior2nd    1460 non-null   object 
     25  MasVnrType     1452 non-null   object 
     26  MasVnrArea     1452 non-null   float64
     27  ExterQual      1460 non-null   object 
     28  ExterCond      1460 non-null   object 
     29  Foundation     1460 non-null   object 
     30  BsmtQual       1423 non-null   object 
     31  BsmtCond       1423 non-null   object 
     32  BsmtExposure   1422 non-null   object 
     33  BsmtFinType1   1423 non-null   object 
     34  BsmtFinSF1     1460 non-null   int64  
     35  BsmtFinType2   1422 non-null   object 
     36  BsmtFinSF2     1460 non-null   int64  
     37  BsmtUnfSF      1460 non-null   int64  
     38  TotalBsmtSF    1460 non-null   int64  
     39  Heating        1460 non-null   object 
     40  HeatingQC      1460 non-null   object 
     41  CentralAir     1460 non-null   object 
     42  Electrical     1459 non-null   object 
     43  1stFlrSF       1460 non-null   int64  
     44  2ndFlrSF       1460 non-null   int64  
     45  LowQualFinSF   1460 non-null   int64  
     46  GrLivArea      1460 non-null   int64  
     47  BsmtFullBath   1460 non-null   int64  
     48  BsmtHalfBath   1460 non-null   int64  
     49  FullBath       1460 non-null   int64  
     50  HalfBath       1460 non-null   int64  
     51  BedroomAbvGr   1460 non-null   int64  
     52  KitchenAbvGr   1460 non-null   int64  
     53  KitchenQual    1460 non-null   object 
     54  TotRmsAbvGrd   1460 non-null   int64  
     55  Functional     1460 non-null   object 
     56  Fireplaces     1460 non-null   int64  
     57  FireplaceQu    770 non-null    object 
     58  GarageType     1379 non-null   object 
     59  GarageYrBlt    1379 non-null   float64
     60  GarageFinish   1379 non-null   object 
     61  GarageCars     1460 non-null   int64  
     62  GarageArea     1460 non-null   int64  
     63  GarageQual     1379 non-null   object 
     64  GarageCond     1379 non-null   object 
     65  PavedDrive     1460 non-null   object 
     66  WoodDeckSF     1460 non-null   int64  
     67  OpenPorchSF    1460 non-null   int64  
     68  EnclosedPorch  1460 non-null   int64  
     69  3SsnPorch      1460 non-null   int64  
     70  ScreenPorch    1460 non-null   int64  
     71  PoolArea       1460 non-null   int64  
     72  PoolQC         7 non-null      object 
     73  Fence          281 non-null    object 
     74  MiscFeature    54 non-null     object 
     75  MiscVal        1460 non-null   int64  
     76  MoSold         1460 non-null   int64  
     77  YrSold         1460 non-null   int64  
     78  SaleType       1460 non-null   object 
     79  SaleCondition  1460 non-null   object 
     80  SalePrice      1460 non-null   int64  
    dtypes: float64(3), int64(35), object(43)
    memory usage: 924.0+ KB



```python
train['Neighborhood'].value_counts()
```




    NAmes      225
    CollgCr    150
    OldTown    113
    Edwards    100
    Somerst     86
    Gilbert     79
    NridgHt     77
    Sawyer      74
    NWAmes      73
    SawyerW     59
    BrkSide     58
    Crawfor     51
    Mitchel     49
    NoRidge     41
    Timber      38
    IDOTRR      37
    ClearCr     28
    StoneBr     25
    SWISU       25
    Blmngtn     17
    MeadowV     17
    BrDale      16
    Veenker     11
    NPkVill      9
    Blueste      2
    Name: Neighborhood, dtype: int64




```python
train['SalePrice'].value_counts()
```




    140000    20
    135000    17
    145000    14
    155000    14
    190000    13
              ..
    84900      1
    424870     1
    415298     1
    62383      1
    34900      1
    Name: SalePrice, Length: 663, dtype: int64




```python
# 결측 데이터 확인
train.isnull().sum()
```




    Id                 0
    MSSubClass         0
    MSZoning           0
    LotFrontage      259
    LotArea            0
                    ... 
    MoSold             0
    YrSold             0
    SaleType           0
    SaleCondition      0
    SalePrice          0
    Length: 81, dtype: int64




```python
# 결측데이터가 한개라도 있는 것들
train.isnull().sum().loc[train.isnull().sum().values>0]
```




    LotFrontage      259
    Alley           1369
    MasVnrType         8
    MasVnrArea         8
    BsmtQual          37
    BsmtCond          37
    BsmtExposure      38
    BsmtFinType1      37
    BsmtFinType2      38
    Electrical         1
    FireplaceQu      690
    GarageType        81
    GarageYrBlt       81
    GarageFinish      81
    GarageQual        81
    GarageCond        81
    PoolQC          1453
    Fence           1179
    MiscFeature     1406
    dtype: int64




```python
# 데이터 자료형에 따라 columns 분류
print(train.dtypes.loc[train.dtypes == 'object'])
print(train.dtypes.loc[train.dtypes == 'int'])
print(train.dtypes.loc[train.dtypes == 'float'])
```

    MSZoning         object
    Street           object
    Alley            object
    LotShape         object
    LandContour      object
    Utilities        object
    LotConfig        object
    LandSlope        object
    Neighborhood     object
    Condition1       object
    Condition2       object
    BldgType         object
    HouseStyle       object
    RoofStyle        object
    RoofMatl         object
    Exterior1st      object
    Exterior2nd      object
    MasVnrType       object
    ExterQual        object
    ExterCond        object
    Foundation       object
    BsmtQual         object
    BsmtCond         object
    BsmtExposure     object
    BsmtFinType1     object
    BsmtFinType2     object
    Heating          object
    HeatingQC        object
    CentralAir       object
    Electrical       object
    KitchenQual      object
    Functional       object
    FireplaceQu      object
    GarageType       object
    GarageFinish     object
    GarageQual       object
    GarageCond       object
    PavedDrive       object
    PoolQC           object
    Fence            object
    MiscFeature      object
    SaleType         object
    SaleCondition    object
    dtype: object
    Id               int64
    MSSubClass       int64
    LotArea          int64
    OverallQual      int64
    OverallCond      int64
    YearBuilt        int64
    YearRemodAdd     int64
    BsmtFinSF1       int64
    BsmtFinSF2       int64
    BsmtUnfSF        int64
    TotalBsmtSF      int64
    1stFlrSF         int64
    2ndFlrSF         int64
    LowQualFinSF     int64
    GrLivArea        int64
    BsmtFullBath     int64
    BsmtHalfBath     int64
    FullBath         int64
    HalfBath         int64
    BedroomAbvGr     int64
    KitchenAbvGr     int64
    TotRmsAbvGrd     int64
    Fireplaces       int64
    GarageCars       int64
    GarageArea       int64
    WoodDeckSF       int64
    OpenPorchSF      int64
    EnclosedPorch    int64
    3SsnPorch        int64
    ScreenPorch      int64
    PoolArea         int64
    MiscVal          int64
    MoSold           int64
    YrSold           int64
    SalePrice        int64
    dtype: object
    LotFrontage    float64
    MasVnrArea     float64
    GarageYrBlt    float64
    dtype: object


## 7. 데이터 분석 및 Feature Engineering
REFERENCE: https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
- Kaggle Leaderboard에 다양한 discussion 자료 참고하여 따라함


```python
train = train_original.copy()
test = test_original.copy()
```


```python
#check the numbers of samples and features
print("The train data size before dropping Id feature is : {} ".format(train.shape))
print("The test data size before dropping Id feature is : {} ".format(test.shape))

#Save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']

#Now drop the 'Id' colum since it's unnecessary for the prediction process.
train.drop("Id", axis =1, inplace=True)
test.drop("Id", axis =1, inplace=True)
#check again the data size after dropping the 'Id' variable
print("\nThe train data size after dropping Id feature is : {} ".format(train.shape)) 
print("The test data size after dropping Id feature is : {} ".format(test.shape))
```

    The train data size before dropping Id feature is : (1460, 81) 
    The test data size before dropping Id feature is : (1459, 80) 
    
    The train data size after dropping Id feature is : (1460, 80) 
    The test data size after dropping Id feature is : (1459, 79) 


### 7-1. 데이터 정리 작업
- Outlier 삭제, 필터링


```python
fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()
```


    
![svg](/assets/house_files/house_21_0.svg)
    


주택 면적이 넓은데 가격이 싼 데이터가 섞여있으므로, 이것은 outlier로 정의하고 삭제 처리


```python
#Deleting outliers
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice'] <300000)].index)

#Check the graphic again
fig, ax = plt.subplots()
ax.scatter(train['GrLivArea'], train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()
```


    
![svg](/assets/house_files/house_23_0.svg)
    


### 7-2. Target Variable
- SalePrice is the variable we need to predict. So let's do some analysis on this variable first.


```python
from scipy import stats
from scipy.stats import norm, skew #for some statistics

sns.distplot(train['SalePrice'] , fit=norm)

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu,sigma))

# Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()
```

    
     mu = 180932.92 and sigma = 79467.79
    



    
![svg](/assets/house_files/house_25_1.svg)
    



    
![svg](/assets/house_files/house_25_2.svg)
    


#### Log-transformation of the target variable


```python
#We use the numpy function log1p which applies log(1+x) to all elements of the column
train['SalePrice'] = np.log1p(train['SalePrice'])

#Check the new distribution
sns.distplot(train['SalePrice'] , fit=norm)

#Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu,sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()
```

    
     mu = 12.02 and sigma = 0.40
    



    
![svg](/assets/house_files/house_27_1.svg)
    



    
![svg](/assets/house_files/house_27_2.svg)
    


- The skew seems now corrected and the data appears more normally distributted
- 'SalePrice`는 이제부터 log(1+x)로 변환하여 사용하겠음


```python
train['SalePrice'].value_counts()
```




    11.849405    20
    11.813037    17
    11.884496    14
    11.951187    14
    12.154785    13
                 ..
    12.091789     1
    12.200562     1
    12.574185     1
    11.198228     1
    11.841423     1
    Name: SalePrice, Length: 662, dtype: int64



### 7-3. Feature engineering
Let's first concatenate the train and test data in the same dataframe


```python
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train['SalePrice'].values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis = 1, inplace=True)
print('all_data size is : {}'.format(all_data.shape))
```

    all_data size is : (2917, 79)


### 7-4. Missing Data 채우기


```python
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
```


```python
all_data_na = all_data_na.drop(all_data_na[all_data_na==0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio': all_data_na})
missing_data.head(20)
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
      <th>Missing Ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>PoolQC</th>
      <td>99.691464</td>
    </tr>
    <tr>
      <th>MiscFeature</th>
      <td>96.400411</td>
    </tr>
    <tr>
      <th>Alley</th>
      <td>93.212204</td>
    </tr>
    <tr>
      <th>Fence</th>
      <td>80.425094</td>
    </tr>
    <tr>
      <th>FireplaceQu</th>
      <td>48.680151</td>
    </tr>
    <tr>
      <th>LotFrontage</th>
      <td>16.660953</td>
    </tr>
    <tr>
      <th>GarageFinish</th>
      <td>5.450806</td>
    </tr>
    <tr>
      <th>GarageYrBlt</th>
      <td>5.450806</td>
    </tr>
    <tr>
      <th>GarageQual</th>
      <td>5.450806</td>
    </tr>
    <tr>
      <th>GarageCond</th>
      <td>5.450806</td>
    </tr>
    <tr>
      <th>GarageType</th>
      <td>5.382242</td>
    </tr>
    <tr>
      <th>BsmtExposure</th>
      <td>2.811107</td>
    </tr>
    <tr>
      <th>BsmtCond</th>
      <td>2.811107</td>
    </tr>
    <tr>
      <th>BsmtQual</th>
      <td>2.776826</td>
    </tr>
    <tr>
      <th>BsmtFinType2</th>
      <td>2.742544</td>
    </tr>
    <tr>
      <th>BsmtFinType1</th>
      <td>2.708262</td>
    </tr>
    <tr>
      <th>MasVnrType</th>
      <td>0.822763</td>
    </tr>
    <tr>
      <th>MasVnrArea</th>
      <td>0.788481</td>
    </tr>
    <tr>
      <th>MSZoning</th>
      <td>0.137127</td>
    </tr>
    <tr>
      <th>BsmtFullBath</th>
      <td>0.068564</td>
    </tr>
  </tbody>
</table>
</div>




```python
f, ax=plt.subplots(figsize=(15,12))
plt.xticks(rotation = '90')
sns.barplot(x=all_data_na.index, y=all_data_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)
```




    Text(0.5, 1.0, 'Percent missing data by feature')




    
![svg](/assets/house_files/house_35_1.svg)
    


#### Data Correlation
- Feature와 SalePrice 와의 상관관계를 살펴보기
- 아래 그래프에서 마지막 row를 보면, 색깔에 따라 양/음의 상관관계 Correlation을 볼 수 있음


```python
#Correlation map to see how features are correlated with SalePrice
corrmat = train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.8, square=True )
```




    <AxesSubplot:>




    
![svg](/assets/house_files/house_37_1.svg)
    


#### Imputing missing values
We impute them by proceeding sequentially through features with missing values
- PoolQC : data description says NA means "No Pool". That make sense, given the huge ratio of missing value (+99%) and majority of houses have no Pool at all in general. 


```python
all_data['PoolQC'] = all_data['PoolQC'].fillna('None')
```

- MiscFeature : data description says NA means "no misc feature"


```python
all_data['MiscFeature'] = all_data['MiscFeature'].fillna('None')
```

- Alley : data descriptin says NA means "no alley access"


```python
all_data['Alley'] = all_data['Alley'].fillna('None')
```

- Fence : data description says NA means 'no fireplace'


```python
all_data["Fence"] = all_data["Fence"].fillna("None")
```

- FireplaceQu : data description says NA means "no fireplace"


```python
all_data['FireplaceQu'] = all_data['FireplaceQu'].fillna('None')
```

- LotFrontage : Since the area of each street connected to the house property most likely have a similar area to other houses in its neighborhood, we can fill in missing values by the median LotFrontage of the neighborhood


```python
#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform('median')
```

- GarageType, GarageFinish, GarageQual and GarageCond : Replacing missing data with None


```python
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')

```

- GarageYrBlt, GarageArea and GarageCars : Replacing missing data with 0 (Since No garage = no cars in such garage.)


```python
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)
```

- BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath : missing values are likely zero for having no basement


```python
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
```

- BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1 and BsmtFinType2 : For all these categorical basement-related features, NaN means that there is no basement


```python
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')
```

- MasVnrArea and MasVnrType : NA most likely means no masonry veneer for these houses. We can fill 0 for the area and None for the type.


```python
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
```

- MSZoning (The general zoning classing) : 'RL' is by far the most common value. So we can fill in missing values with 'RL'


```python
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
```

- Utilities : For this categorical feature all records are "AllPub", except for one "NoSeWa" and  2 NA. Since the house with 'NoSewa' is the training set, **this feature won't help in predictive modeling**. We can then safely remove it.


```python
all_data = all_data.drop(['Utilities'], axis=1)
```

- Functional : data description says NA means typical


```python
all_data["Functional"] = all_data["Functional"].fillna(all_data["Functional"].mode()[0])
```

- Electrical: It has one NA value. Since this feature has mostly 'SBkr', we can set that for the missing value


```python
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
```

- KitchenQual : Only one NA value, ans same as Electrical, we set 'TA' (which is the most frequent) for the missing value in KitchenQual.


```python
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0]) 
```

- Exterior1st and Exterior2nd : Again Both Exterior 1 & 2 have only one missing value. We will just substitute in the most common string


```python
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
```

- SaleType : Fill in again with most frequent which is "WD"


```python
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
```

- MSSubClass : Na most likely means No building class. We can replace missing values with None


```python
all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")
```

Is there any remaining missing value ?


```python
#Check remaining missing values if any
all_data_na = (all_data.isnull().sum() / len(all_data))*100
all_data_na = all_data_na.drop(all_data_na[all_data_na==0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio': all_data_na})
missing_data.head()
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
      <th>Missing Ratio</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



It remains no missing value.

### 7-5. More features engineering
**Transforming some numerical variables that are really categorical**


```python
#MSSubClass = The building class
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

#Changing OverallCond into a categorical variable
all_data['OverallCond'] = all_data['OverallCond'].astype(str)

#Year and month sold are transformed into categorical features
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)

```

#### Label Encoding some categorical variables that may contain information in their ordering set


```python
from sklearn.preprocessing import LabelEncoder
```


```python
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
    
#process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder()
    lbl.fit(list(all_data[c].values))
    all_data[c] = lbl.transform(list(all_data[c].values))
help(LabelEncoder)
#LabelEncoder란 문자로 되어있는 class 를 숫자로 매핑시켜주는 역할을 함
```

    Help on class LabelEncoder in module sklearn.preprocessing._label:
    
    class LabelEncoder(sklearn.base.TransformerMixin, sklearn.base.BaseEstimator)
     |  Encode target labels with value between 0 and n_classes-1.
     |  
     |  This transformer should be used to encode target values, *i.e.* `y`, and
     |  not the input `X`.
     |  
     |  Read more in the :ref:`User Guide <preprocessing_targets>`.
     |  
     |  .. versionadded:: 0.12
     |  
     |  Attributes
     |  ----------
     |  classes_ : array of shape (n_class,)
     |      Holds the label for each class.
     |  
     |  Examples
     |  --------
     |  `LabelEncoder` can be used to normalize labels.
     |  
     |  >>> from sklearn import preprocessing
     |  >>> le = preprocessing.LabelEncoder()
     |  >>> le.fit([1, 2, 2, 6])
     |  LabelEncoder()
     |  >>> le.classes_
     |  array([1, 2, 6])
     |  >>> le.transform([1, 1, 2, 6])
     |  array([0, 0, 1, 2]...)
     |  >>> le.inverse_transform([0, 0, 1, 2])
     |  array([1, 1, 2, 6])
     |  
     |  It can also be used to transform non-numerical labels (as long as they are
     |  hashable and comparable) to numerical labels.
     |  
     |  >>> le = preprocessing.LabelEncoder()
     |  >>> le.fit(["paris", "paris", "tokyo", "amsterdam"])
     |  LabelEncoder()
     |  >>> list(le.classes_)
     |  ['amsterdam', 'paris', 'tokyo']
     |  >>> le.transform(["tokyo", "tokyo", "paris"])
     |  array([2, 2, 1]...)
     |  >>> list(le.inverse_transform([2, 2, 1]))
     |  ['tokyo', 'tokyo', 'paris']
     |  
     |  See also
     |  --------
     |  sklearn.preprocessing.OrdinalEncoder : Encode categorical features
     |      using an ordinal encoding scheme.
     |  
     |  sklearn.preprocessing.OneHotEncoder : Encode categorical features
     |      as a one-hot numeric array.
     |  
     |  Method resolution order:
     |      LabelEncoder
     |      sklearn.base.TransformerMixin
     |      sklearn.base.BaseEstimator
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  fit(self, y)
     |      Fit label encoder
     |      
     |      Parameters
     |      ----------
     |      y : array-like of shape (n_samples,)
     |          Target values.
     |      
     |      Returns
     |      -------
     |      self : returns an instance of self.
     |  
     |  fit_transform(self, y)
     |      Fit label encoder and return encoded labels
     |      
     |      Parameters
     |      ----------
     |      y : array-like of shape [n_samples]
     |          Target values.
     |      
     |      Returns
     |      -------
     |      y : array-like of shape [n_samples]
     |  
     |  inverse_transform(self, y)
     |      Transform labels back to original encoding.
     |      
     |      Parameters
     |      ----------
     |      y : numpy array of shape [n_samples]
     |          Target values.
     |      
     |      Returns
     |      -------
     |      y : numpy array of shape [n_samples]
     |  
     |  transform(self, y)
     |      Transform labels to normalized encoding.
     |      
     |      Parameters
     |      ----------
     |      y : array-like of shape [n_samples]
     |          Target values.
     |      
     |      Returns
     |      -------
     |      y : array-like of shape [n_samples]
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from sklearn.base.TransformerMixin:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from sklearn.base.BaseEstimator:
     |  
     |  __getstate__(self)
     |  
     |  __repr__(self, N_CHAR_MAX=700)
     |      Return repr(self).
     |  
     |  __setstate__(self, state)
     |  
     |  get_params(self, deep=True)
     |      Get parameters for this estimator.
     |      
     |      Parameters
     |      ----------
     |      deep : bool, default=True
     |          If True, will return the parameters for this estimator and
     |          contained subobjects that are estimators.
     |      
     |      Returns
     |      -------
     |      params : mapping of string to any
     |          Parameter names mapped to their values.
     |  
     |  set_params(self, **params)
     |      Set the parameters of this estimator.
     |      
     |      The method works on simple estimators as well as on nested objects
     |      (such as pipelines). The latter have parameters of the form
     |      ``<component>__<parameter>`` so that it's possible to update each
     |      component of a nested object.
     |      
     |      Parameters
     |      ----------
     |      **params : dict
     |          Estimator parameters.
     |      
     |      Returns
     |      -------
     |      self : object
     |          Estimator instance.
    



```python
#LabelEncoder 결과 샘플
all_data['FireplaceQu'].head()
```




    0    3
    1    5
    2    5
    3    2
    4    5
    Name: FireplaceQu, dtype: int64




```python
# shape        
print('Shape all_data: {}'.format(all_data.shape))
```

    Shape all_data: (2917, 78)


#### Adding one more important feature
- Since area related features are very important to determine house prices, we add one more feature which is the total area of basement, first and second floor areas of each house


```python
# Adding total sqfootage feature
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
```

#### Skewed features


```python
numeric_feats = all_data.dtypes[all_data.dtypes != 'object'].index

# Check the skew of all numerical features
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
```


```python
help(skew)
#skew 함수는 노말디스트리뷰션 성질을 해당 데이터가 가지고 있는지 지수로 보여주는 함수인 것이고, 0에 가까울 수록 좋다.
#ascending = False 라는 것은 내림차순
```

    Help on function skew in module scipy.stats.stats:
    
    skew(a, axis=0, bias=True, nan_policy='propagate')
        Compute the sample skewness of a data set.
        
        For normally distributed data, the skewness should be about zero. For
        unimodal continuous distributions, a skewness value greater than zero means
        that there is more weight in the right tail of the distribution. The
        function `skewtest` can be used to determine if the skewness value
        is close enough to zero, statistically speaking.
        
        Parameters
        ----------
        a : ndarray
            Input array.
        axis : int or None, optional
            Axis along which skewness is calculated. Default is 0.
            If None, compute over the whole array `a`.
        bias : bool, optional
            If False, then the calculations are corrected for statistical bias.
        nan_policy : {'propagate', 'raise', 'omit'}, optional
            Defines how to handle when input contains nan.
            The following options are available (default is 'propagate'):
        
              * 'propagate': returns nan
              * 'raise': throws an error
              * 'omit': performs the calculations ignoring nan values
        
        Returns
        -------
        skewness : ndarray
            The skewness of values along an axis, returning 0 where all values are
            equal.
        
        Notes
        -----
        The sample skewness is computed as the Fisher-Pearson coefficient
        of skewness, i.e.
        
        .. math::
        
            g_1=\frac{m_3}{m_2^{3/2}}
        
        where
        
        .. math::
        
            m_i=\frac{1}{N}\sum_{n=1}^N(x[n]-\bar{x})^i
        
        is the biased sample :math:`i\texttt{th}` central moment, and :math:`\bar{x}` is
        the sample mean.  If ``bias`` is False, the calculations are
        corrected for bias and the value computed is the adjusted
        Fisher-Pearson standardized moment coefficient, i.e.
        
        .. math::
        
            G_1=\frac{k_3}{k_2^{3/2}}=
                \frac{\sqrt{N(N-1)}}{N-2}\frac{m_3}{m_2^{3/2}}.
        
        References
        ----------
        .. [1] Zwillinger, D. and Kokoska, S. (2000). CRC Standard
           Probability and Statistics Tables and Formulae. Chapman & Hall: New
           York. 2000.
           Section 2.2.24.1
        
        Examples
        --------
        >>> from scipy.stats import skew
        >>> skew([1, 2, 3, 4, 5])
        0.0
        >>> skew([2, 8, 0, 4, 1, 9, 9, 0])
        0.2650554122698573
    



```python
print('\nSkew in numerical features: \n')
skewness = pd.DataFrame({'Skew' : skewed_feats})
skewness.head(10)
```

    
    Skew in numerical features: 
    





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
      <th>Skew</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>MiscVal</th>
      <td>21.939672</td>
    </tr>
    <tr>
      <th>PoolArea</th>
      <td>17.688664</td>
    </tr>
    <tr>
      <th>LotArea</th>
      <td>13.109495</td>
    </tr>
    <tr>
      <th>LowQualFinSF</th>
      <td>12.084539</td>
    </tr>
    <tr>
      <th>3SsnPorch</th>
      <td>11.372080</td>
    </tr>
    <tr>
      <th>LandSlope</th>
      <td>4.973254</td>
    </tr>
    <tr>
      <th>KitchenAbvGr</th>
      <td>4.300550</td>
    </tr>
    <tr>
      <th>BsmtFinSF2</th>
      <td>4.144503</td>
    </tr>
    <tr>
      <th>EnclosedPorch</th>
      <td>4.002344</td>
    </tr>
    <tr>
      <th>ScreenPorch</th>
      <td>3.945101</td>
    </tr>
  </tbody>
</table>
</div>



### 7-6. Box Cox Transformation of (highly) skewed features

- We use the scipy function boxcox1p which computes the Box-Cox transformation of 1+x.
 Note that settign $\lambda$ = 0 is equivalent to log1p used above for the target variable.


```python
skweness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    all_data[feat] = boxcox1p(all_data[feat], lam)
```

    There are 59 skewed numerical features to Box Cox transform


### 7-7. Getting dummy categorical features


```python
help(pd.get_dummies)
```

    Help on function get_dummies in module pandas.core.reshape.reshape:
    
    get_dummies(data, prefix=None, prefix_sep='_', dummy_na=False, columns=None, sparse=False, drop_first=False, dtype=None) -> 'DataFrame'
        Convert categorical variable into dummy/indicator variables.
        
        Parameters
        ----------
        data : array-like, Series, or DataFrame
            Data of which to get dummy indicators.
        prefix : str, list of str, or dict of str, default None
            String to append DataFrame column names.
            Pass a list with length equal to the number of columns
            when calling get_dummies on a DataFrame. Alternatively, `prefix`
            can be a dictionary mapping column names to prefixes.
        prefix_sep : str, default '_'
            If appending prefix, separator/delimiter to use. Or pass a
            list or dictionary as with `prefix`.
        dummy_na : bool, default False
            Add a column to indicate NaNs, if False NaNs are ignored.
        columns : list-like, default None
            Column names in the DataFrame to be encoded.
            If `columns` is None then all the columns with
            `object` or `category` dtype will be converted.
        sparse : bool, default False
            Whether the dummy-encoded columns should be backed by
            a :class:`SparseArray` (True) or a regular NumPy array (False).
        drop_first : bool, default False
            Whether to get k-1 dummies out of k categorical levels by removing the
            first level.
        dtype : dtype, default np.uint8
            Data type for new columns. Only a single dtype is allowed.
        
            .. versionadded:: 0.23.0
        
        Returns
        -------
        DataFrame
            Dummy-coded data.
        
        See Also
        --------
        Series.str.get_dummies : Convert Series to dummy codes.
        
        Examples
        --------
        >>> s = pd.Series(list('abca'))
        
        >>> pd.get_dummies(s)
           a  b  c
        0  1  0  0
        1  0  1  0
        2  0  0  1
        3  1  0  0
        
        >>> s1 = ['a', 'b', np.nan]
        
        >>> pd.get_dummies(s1)
           a  b
        0  1  0
        1  0  1
        2  0  0
        
        >>> pd.get_dummies(s1, dummy_na=True)
           a  b  NaN
        0  1  0    0
        1  0  1    0
        2  0  0    1
        
        >>> df = pd.DataFrame({'A': ['a', 'b', 'a'], 'B': ['b', 'a', 'c'],
        ...                    'C': [1, 2, 3]})
        
        >>> pd.get_dummies(df, prefix=['col1', 'col2'])
           C  col1_a  col1_b  col2_a  col2_b  col2_c
        0  1       1       0       0       1       0
        1  2       0       1       1       0       0
        2  3       1       0       0       0       1
        
        >>> pd.get_dummies(pd.Series(list('abcaa')))
           a  b  c
        0  1  0  0
        1  0  1  0
        2  0  0  1
        3  1  0  0
        4  1  0  0
        
        >>> pd.get_dummies(pd.Series(list('abcaa')), drop_first=True)
           b  c
        0  0  0
        1  1  0
        2  0  1
        3  0  0
        4  0  0
        
        >>> pd.get_dummies(pd.Series(list('abc')), dtype=float)
             a    b    c
        0  1.0  0.0  0.0
        1  0.0  1.0  0.0
        2  0.0  0.0  1.0
    



```python
all_data = pd.get_dummies(all_data)
print(all_data.shape)
all_data.columns
```

    (2917, 220)





    Index(['MSSubClass', 'LotFrontage', 'LotArea', 'Street', 'Alley', 'LotShape',
           'LandSlope', 'OverallQual', 'OverallCond', 'YearBuilt',
           ...
           'SaleType_ConLw', 'SaleType_New', 'SaleType_Oth', 'SaleType_WD',
           'SaleCondition_Abnorml', 'SaleCondition_AdjLand',
           'SaleCondition_Alloca', 'SaleCondition_Family', 'SaleCondition_Normal',
           'SaleCondition_Partial'],
          dtype='object', length=220)



- Getting the new train and test sets.


```python
train = all_data[:ntrain]
test = all_data[ntrain:]
```

## 8. Modeling
- **Import libraries**


```python
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
```

### 8-1. Define a cross validation strategy
We use the **cross_val_score** function of Sklearn. However this function has not a shuffle attribute, we add then one line of code, in order to shuffle the dataset prior to cross-validation


```python
#Validation function
n_folds=5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model, train.values, y_train, scoring='neg_mean_squared_error', cv=kf))
    return rmse
# 참고로 logistic prediction 을 할 때에는 scoring 인자로 'accuracy'를 전달했었음: 타이타닉 문제
'''
 k_fold = KFold(n_splits = 10, shuffle = True, random_state = 0)
 score = cross_val_score(clf, train_data, target, cv= k_fold, n_jobs =1 , scoring='accuracy')
'''
```




    "\n k_fold = KFold(n_splits = 10, shuffle = True, random_state = 0)\n score = cross_val_score(clf, train_data, target, cv= k_fold, n_jobs =1 , scoring='accuracy')\n"




```python
help(cross_val_score)
```

    Help on function cross_val_score in module sklearn.model_selection._validation:
    
    cross_val_score(estimator, X, y=None, *, groups=None, scoring=None, cv=None, n_jobs=None, verbose=0, fit_params=None, pre_dispatch='2*n_jobs', error_score=nan)
        Evaluate a score by cross-validation
        
        Read more in the :ref:`User Guide <cross_validation>`.
        
        Parameters
        ----------
        estimator : estimator object implementing 'fit'
            The object to use to fit the data.
        
        X : array-like of shape (n_samples, n_features)
            The data to fit. Can be for example a list, or an array.
        
        y : array-like of shape (n_samples,) or (n_samples, n_outputs),             default=None
            The target variable to try to predict in the case of
            supervised learning.
        
        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set. Only used in conjunction with a "Group" :term:`cv`
            instance (e.g., :class:`GroupKFold`).
        
        scoring : str or callable, default=None
            A str (see model evaluation documentation) or
            a scorer callable object / function with signature
            ``scorer(estimator, X, y)`` which should return only
            a single value.
        
            Similar to :func:`cross_validate`
            but only a single metric is permitted.
        
            If None, the estimator's default scorer (if available) is used.
        
        cv : int, cross-validation generator or an iterable, default=None
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:
        
            - None, to use the default 5-fold cross validation,
            - int, to specify the number of folds in a `(Stratified)KFold`,
            - :term:`CV splitter`,
            - An iterable yielding (train, test) splits as arrays of indices.
        
            For int/None inputs, if the estimator is a classifier and ``y`` is
            either binary or multiclass, :class:`StratifiedKFold` is used. In all
            other cases, :class:`KFold` is used.
        
            Refer :ref:`User Guide <cross_validation>` for the various
            cross-validation strategies that can be used here.
        
            .. versionchanged:: 0.22
                ``cv`` default value if None changed from 3-fold to 5-fold.
        
        n_jobs : int, default=None
            The number of CPUs to use to do the computation.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.
        
        verbose : int, default=0
            The verbosity level.
        
        fit_params : dict, default=None
            Parameters to pass to the fit method of the estimator.
        
        pre_dispatch : int or str, default='2*n_jobs'
            Controls the number of jobs that get dispatched during parallel
            execution. Reducing this number can be useful to avoid an
            explosion of memory consumption when more jobs get dispatched
            than CPUs can process. This parameter can be:
        
                - None, in which case all the jobs are immediately
                  created and spawned. Use this for lightweight and
                  fast-running jobs, to avoid delays due to on-demand
                  spawning of the jobs
        
                - An int, giving the exact number of total jobs that are
                  spawned
        
                - A str, giving an expression as a function of n_jobs,
                  as in '2*n_jobs'
        
        error_score : 'raise' or numeric, default=np.nan
            Value to assign to the score if an error occurs in estimator fitting.
            If set to 'raise', the error is raised.
            If a numeric value is given, FitFailedWarning is raised. This parameter
            does not affect the refit step, which will always raise the error.
        
            .. versionadded:: 0.20
        
        Returns
        -------
        scores : array of float, shape=(len(list(cv)),)
            Array of scores of the estimator for each run of the cross validation.
        
        Examples
        --------
        >>> from sklearn import datasets, linear_model
        >>> from sklearn.model_selection import cross_val_score
        >>> diabetes = datasets.load_diabetes()
        >>> X = diabetes.data[:150]
        >>> y = diabetes.target[:150]
        >>> lasso = linear_model.Lasso()
        >>> print(cross_val_score(lasso, X, y, cv=3))
        [0.33150734 0.08022311 0.03531764]
        
        See Also
        ---------
        :func:`sklearn.model_selection.cross_validate`:
            To run cross-validation on multiple metrics and also to return
            train scores, fit times and score times.
        
        :func:`sklearn.model_selection.cross_val_predict`:
            Get predictions from each split of cross-validation for diagnostic
            purposes.
        
        :func:`sklearn.metrics.make_scorer`:
            Make a scorer from a performance metric or loss function.
    


- make_pipeline 함수: [설명](https://skasha.tistory.com/80)  
![image](https://t1.daumcdn.net/cfile/tistory/99EF85365E24E3A41A)

### 8-2. Base models
- LASSO Regression:
    This model may be very sensitive to outliers. So we need to made it more robust on them. For that we use the sklearn's Robustscaler() method on pipeline


```python
lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))
```

- Elastic Net Regression:
    again made robust to outliers


```python
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=0.9, random_state=3))
```

- Kernel Ridge Regression:


```python
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
```

- Gradient Boosting Regression:
With huber loss that makes it robust to outliers


```python
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, loss='huber', random_state=5)
```

- XGBoost


```python
model_xgb = xgb.XGBRegressor(colsample_bytree = 0.4603, gamma=0.0468, learning_rate = 0.05, max_depth=3,min_child_weight=1.7817, n_estimators=2200, reg_alpha=0.4640, reg_lambda=0.8571, subsample=0.5213, silent=1, random_state=7, nthread=-1)
```

- LightGBM:


```python
model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=5, learning_rage =0.05, n_estimators=720, max_bin=55, bagging_fraction=0.8, bagging_freq=5, feature_fraction=0.2319, feature_fraction_seed=9, bagging_seed=9, min_data_in_leaf = 6, min_sum_hessian_in_leaf=11)
```

### 8-3. Base Models scores
Let's see how these base models perform on the data by evaluating the cross-validation rmsle error


```python
score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
```

    
    Lasso score: 0.1116 (0.0075)
    



```python
score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
```

    ElasticNet score: 0.1116 (0.0075)
    



```python
score = rmsle_cv(KRR)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
```

    Kernel Ridge score: 0.1153 (0.0077)
    



```python
score = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
```

    Gradient Boosting score: 0.1163 (0.0084)
    



```python
score = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
```

    [01:04:27] WARNING: /tmp/build/80754af9/xgboost-split_1619724447847/work/src/learner.cc:541: 
    Parameters: { silent } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    
    [01:04:34] WARNING: /tmp/build/80754af9/xgboost-split_1619724447847/work/src/learner.cc:541: 
    Parameters: { silent } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    
    [01:04:41] WARNING: /tmp/build/80754af9/xgboost-split_1619724447847/work/src/learner.cc:541: 
    Parameters: { silent } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    
    [01:04:50] WARNING: /tmp/build/80754af9/xgboost-split_1619724447847/work/src/learner.cc:541: 
    Parameters: { silent } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    
    [01:04:56] WARNING: /tmp/build/80754af9/xgboost-split_1619724447847/work/src/learner.cc:541: 
    Parameters: { silent } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    
    Xgboost score: 0.1160 (0.0052)
    



```python
score = rmsle_cv(model_lgb)
print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))
```

    [LightGBM] [Warning] Unknown parameter: learning_rage
    [LightGBM] [Warning] feature_fraction is set=0.2319, colsample_bytree=1.0 will be ignored. Current value: feature_fraction=0.2319
    [LightGBM] [Warning] min_sum_hessian_in_leaf is set=11, min_child_weight=0.001 will be ignored. Current value: min_sum_hessian_in_leaf=11
    [LightGBM] [Warning] min_data_in_leaf is set=6, min_child_samples=20 will be ignored. Current value: min_data_in_leaf=6
    [LightGBM] [Warning] bagging_freq is set=5, subsample_freq=0 will be ignored. Current value: bagging_freq=5
    [LightGBM] [Warning] bagging_fraction is set=0.8, subsample=1.0 will be ignored. Current value: bagging_fraction=0.8
    [LightGBM] [Warning] Unknown parameter: learning_rage
    [LightGBM] [Warning] feature_fraction is set=0.2319, colsample_bytree=1.0 will be ignored. Current value: feature_fraction=0.2319
    [LightGBM] [Warning] min_sum_hessian_in_leaf is set=11, min_child_weight=0.001 will be ignored. Current value: min_sum_hessian_in_leaf=11
    [LightGBM] [Warning] min_data_in_leaf is set=6, min_child_samples=20 will be ignored. Current value: min_data_in_leaf=6
    [LightGBM] [Warning] bagging_freq is set=5, subsample_freq=0 will be ignored. Current value: bagging_freq=5
    [LightGBM] [Warning] bagging_fraction is set=0.8, subsample=1.0 will be ignored. Current value: bagging_fraction=0.8
    [LightGBM] [Warning] Unknown parameter: learning_rage
    [LightGBM] [Warning] feature_fraction is set=0.2319, colsample_bytree=1.0 will be ignored. Current value: feature_fraction=0.2319
    [LightGBM] [Warning] min_sum_hessian_in_leaf is set=11, min_child_weight=0.001 will be ignored. Current value: min_sum_hessian_in_leaf=11
    [LightGBM] [Warning] min_data_in_leaf is set=6, min_child_samples=20 will be ignored. Current value: min_data_in_leaf=6
    [LightGBM] [Warning] bagging_freq is set=5, subsample_freq=0 will be ignored. Current value: bagging_freq=5
    [LightGBM] [Warning] bagging_fraction is set=0.8, subsample=1.0 will be ignored. Current value: bagging_fraction=0.8
    [LightGBM] [Warning] Unknown parameter: learning_rage
    [LightGBM] [Warning] feature_fraction is set=0.2319, colsample_bytree=1.0 will be ignored. Current value: feature_fraction=0.2319
    [LightGBM] [Warning] min_sum_hessian_in_leaf is set=11, min_child_weight=0.001 will be ignored. Current value: min_sum_hessian_in_leaf=11
    [LightGBM] [Warning] min_data_in_leaf is set=6, min_child_samples=20 will be ignored. Current value: min_data_in_leaf=6
    [LightGBM] [Warning] bagging_freq is set=5, subsample_freq=0 will be ignored. Current value: bagging_freq=5
    [LightGBM] [Warning] bagging_fraction is set=0.8, subsample=1.0 will be ignored. Current value: bagging_fraction=0.8
    [LightGBM] [Warning] Unknown parameter: learning_rage
    [LightGBM] [Warning] feature_fraction is set=0.2319, colsample_bytree=1.0 will be ignored. Current value: feature_fraction=0.2319
    [LightGBM] [Warning] min_sum_hessian_in_leaf is set=11, min_child_weight=0.001 will be ignored. Current value: min_sum_hessian_in_leaf=11
    [LightGBM] [Warning] min_data_in_leaf is set=6, min_child_samples=20 will be ignored. Current value: min_data_in_leaf=6
    [LightGBM] [Warning] bagging_freq is set=5, subsample_freq=0 will be ignored. Current value: bagging_freq=5
    [LightGBM] [Warning] bagging_fraction is set=0.8, subsample=1.0 will be ignored. Current value: bagging_fraction=0.8
    LGBM score: 0.1199 (0.0078)
    


## 9. Stacking models
**Simplest Stacking approach : Averaging base models**  
We begin with this simple approach of averaging base models. We build a new class to extend scikit-learn with our model and also to laverage encapsulation and code reuse (inheritance)

### 9-1. Averaged base models class
