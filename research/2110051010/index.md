---
layout: post
type: research
date: 2021-10-05 10:10
category: Localization
title: Localization- Essential with Bayes Filter
subtitle: 1D Localization Problem with bayes
writer: 100
post-header: true
header-img: img/localization_intuition.png
hash-tag: [Localization, Baysian, Filter]
use_math: true
---

# Localization Essential 
> AUTHOR: Sungwook LE    
> DATE: '21.10/5  

## 1. Introduction
- Localization Insight: check below image..[![Intuition](/assets/localization_intuition.png)](https://youtu.be/U-uDtVgezcE)
- Localization needs informations such as `MAP, Onboard Sensor, GPS...`    
- Filters that could be used are `Histogram Filters, Kalman Filters, Particle Filters...`   

> * Localization:  
        1. "Where is our car in a given map with an accuracy of 10cm or less?"  
        2. "Onboard Sensors are used to estimate transformation between **measurements** and a **given map**."  

## 2. 로봇 1차원 좌표 localization
- Bayes Rule에 기반한 로봇의 1차원 Localization 문제에서의 Update단계는 아래 그림과 같다.  
![1D](/assets/Localization_1D.png)

- Bayes Rule 수식으로 위의 상황을 설명해보자   
[![image](/assets/bayes_rule.png)](https://youtu.be/sA5wv56qYc0)
        - Posterior는 $P(X_i|Z)$이고, Prior는 $P(X_i)$ = 0.2이다. 
        - Measurement가 빨간색이었다고 하면, 빨간색 cell에는 0.6을 곱하고 그렇지 않은 cell에는 0.2를 곱한다고 하자. 이게 $P(Z|X_i)$ 즉, 관측됬을 때 실제 그 위치에 있을 확률이 된다.
        - Prior $P(X_i)$와 $P(Z|X_i)$를 곱하고 전체 확률 (P(Z))로 Normalization을 해준 값이 Bayes Rule 업데이트가 된다.

- 좀 더 확장해서 표현 (추론 이론에 접목)  
![image](/assets/bayes_inference.png)
P(X) 부분이 모델이라고 표현된 부분이 잇는데 이 부분은 추론이 `predict와 update`로 구성되니까 P(X)는 `predict`단계에서 넘어온 것이고 `predict`는 모델에 관한 함수여서 그런 것이다.


- 책에서 쉽게 표현된 그림으로 설명하면 데이터가 관측값이 되고 이 값이 사전 믿음에 곱해지면 업데이트가 되는 지극히 상식적이고 쉬운 과정을 확률로서 표현한 것 뿐이다.
![image](https://mblogthumb-phinf.pstatic.net/MjAyMDA1MDNfMzgg/MDAxNTg4NTEwNjAxNDUz.ml8si80x40eByFDGNQpQDPd1laT4z3U2Mwzmvxr8MTEg.URdec5gyQaB5IqJL0FY-vwbUENJSvSgW6Tzari9AMMEg.PNG.souhaits9/image.png?type=w800)

- Model Process에서는 불확실성이 더해지는 형태가 되고, forward 연산이 수행된다.
![image](/assets/predict_update_step.png)

- In general, entropy represents the amount of uncertainty in a system.

* 용어:
    - `BELIEF` = Probability
    - `Sense(update)` = Measurement (Product, followed by Normalization)
    - `Move(predict)` =  Convolution(=Adding)
            - 왜 convolution이라 표현했냐면, 여러 파티클에 대해 각각 move가 적용되어 predict 되어야 하기 때문이다.

## 3. Summary Localization
[![image](/assets/localization_summary.png)](https://youtu.be/WCva9DtGgGA)
    1. 주어진 맵에서 내 위치를 찾는 것 (`Local Localization`)  
    2. 주어진 맵과 `Global Map`간의 transformation 관계를 안다면, `Global Localization` 까지 가능  
    3. 위 그림에서 $bel(X_t) = P(X_t|z_{1:t}, u_{1:t}, m)$ 된다.
    - 번외로, SLAM(Simultaneously Localization And Mapping)에서는 Map까지 작성을 해야하니,
    $P(x_t, m|z_{1:t}, u_{1:t})$가 된다.  

- 전체 프로세스 (`Bayes Filter`)
[![image](/assets/bayes_process.png)](https://youtu.be/teVw2J-_6ZE)
- 상태 추정 문제에서는 관심을 갖고 있는 State에 대한 값을 계속 업데이트 해 간 것이고
- 측위 문제는 모든 관측값에 대한 보정된 확률을 다 가져와서(곱해서), 여러 파티클 중 가장 **매칭** 확률이 높은 파티클을 현재 위치로 측위하게 되는 것이다.
- 강의 자료 [참고](https://classroom.udacity.com/nanodegrees/nd013/parts/b9040951-b43f-4dd3-8b16-76e7b52f4d9d/modules/85ece059-1351-4599-bb2c-0095d6534c8c/lessons/2ac1492e-9320-4e42-91a5-0845e4f77b0c/concepts/3967f970-584e-4fcd-9708-677f9b5f43f9)

- **Bayes Filter for Localization(Markov Localization)**  
$bel(x_t) = p(x_t|z_t,z_{1:t-1}, u_{1:t}, m)=\eta * p(z_t|x_t,m)\hat{bel}(x_t)$

- `Markov Localization`, `Kalman Filters` 그리고 `Particle Filters`는 `Bayes Filter`의 `Realization` **표현형**이다.


## 4. 1D Localization uisng Bayesian Rule Practice
- bayes 이론을 그대로 접목하여 로컬라이제이션 필터를 만들었다. 
- 1D given map에서 푸는 문제였기 때문에 주어진 모든 경우의 수에 대하여 navie하게 **전부**를 계산하여 접근하였기 때문에 `bayes` 그 자체를 사용할 수 있었다.
- Code is Here: [MyRepo](https://github.com/SungwookLE/Codingtest_Baekjoon/blob/master/localization_1d.cpp)  


## 끝
