---
title:  "Kalman Filter Essential: Linear"
excerpt: "선형 칼만필터를 이해하고 유도해보자"

categories:
  - research
tags:
  - research

toc: true
toc_sticky: true
 
date: 2021-09-12
---
# Linear Kalman Filter Essential
AUTHOR: SungwookLE  
DATE: '21.9/12  
LECTURE: [Udacity Kalman Filter](https://classroom.udacity.com/nanodegrees/nd013/parts/168c60f1-cc92-450a-a91b-e427c326e6a7/modules/95d62426-4da9-49a6-9195-603e0f81d3f1/lessons/4a76ef9b-27d7-4aa4-8f3c-fd2b1e8757a4/concepts/487410280923)  

## 1. 칼만 필터
칼만필터는 너무 유명해서, 제어 분야에서 모르는 사람은 없을 정도다. 그만큼 사람들이 정리해 둔 글도 많지만, 볼 때마다 깨닫는 부분이 있는 것을 보면 아직도 모자라기만 한 듯하여 내 식대로, 단순하고 효과적인 칼만필터에 대해서 정리해보고자한다.  
![image](https://miro.medium.com/max/1400/1*oQ72mFforQ_zt9GuwTykXA.png)  

위 그림과 같이 칼만필터는 예측단계(Prediction, Model Forward)와 측정값을 가지고 보정하는 보정 단계로 이루어져 있다.(Measuremnet Update, Correction)
위 그림의 수식을 잘 살펴보면 측정 단계의 분산과 예측단계의 분산을 weighting 인자로 하여 평균을 구하고 그 평균을 구하는 것을 볼 수 있는데, 이 수식이야 말로 확률 이론에서 유도되는 칼만필터의 진 면목을 보여주는 수식이라 생각이 들었다.

물론, 칼만필터를 유도하는 방식은 여러가지가 있는데, 목적함수를 최소화하는 최적 게인 K를 구하는 방식으로 유도하는 방식이 있고(해당 방식이 내가 익숙하다고 여겼던 방식), Orthogonal Principle 을 이용하여 내적 = 0 의 관계를 이용하여 유도하는 방식이 있고 확률에서 유도하는 방식이 있다. 아무래도 수식으로만 쳐다보고 있으면 목적함수의 1차 미분은 0이 되는 게인 K를 구하는 방식이 제일 이해하기 편안하나, 확률 이론의 진면목을 살펴보는 것에 한계가 있기에 이번 기회에 확률 이론으로 이해하는 방식으로 스터디를 해보았다.

결론 부터 말하자면,
예측 단계는 Total Probability Theory (전확률) 법칙으로 유도하고
보정 단계는 Bayes Rule을 이용하여 업데이트 되는 과정이다.



