---
title:  "Localization: Particle Filter"
excerpt: "Particle Filter Study in Localization"

categories:
  - research
tags:
  - research

toc: true
toc_sticky: true
use_math: true
 
date: 2021-10-07
---

# Localization: Particle Filter 
> AUTHOR: Sungwook LE    
> DATE: '21.10/7  

## 1. Introduction

- 파티클 필터를 이용하여 localization을 문제를 풀 수 있다.
- localization 문제에서 `Kalman Filter`는 효율적이나, `uni-modal`의 **belif**를 가지고 있다는 것이 큰 단점이 된다.
- `Particle Filter`는 particle의 개수에 따라 효율이 결정되지만, `multi-modal`을 풀 수 있다는 장점이 있다.
- `multi-modal`을 풀 수 있다는 것은 *highly non-linear* (예를 들면, irregular 공간 점프) 등의 상황에서도 localization을 풀 수 있는 장점이 있다.

## 2. Particle Filter Basic
- 구현은 쉬운 편인데, 파티클 여러개를 MAP위에 생성하고, `LandMark`와 `measurement`의 *Gausian Matching* 확률 정보를 Weight로 하여 값을 기준으로 `Resampling`하는 과정을 통해 매칭 확률이 높은 Particle이 생존하게 되는 방식이다.

- 리샘플링은 `weight`가 클수록 더 높은 확률롤 뽑히게끔 만들어주면 됨: `resampling wheel`이라는 것을 이용할 수도 있다
    - wheel의 둘레를 $\beta = random * Weight_{max}$ 로 선언하고 $index_{init} = random$으로 하여, 
    ```python
    for i in range(Particle 개수:)
        while (beta < Weight[index]):
                beta -= Weight[index]
                index=(index+1)%N
        Pick.append(Particle[index])
    ```
    하는 형태이다.

- 파티클 필터는 particle들을 랜덤하게 여러군데 뿌린다음에 각각의 방향으로 move(`predict`)하고, `observation`과 `landmark`의 matching 정보를 확률로 계산하고 이 것을 weight의 가중치로 하여 resample 하는 과정으로 `bayesian filter`의 realization의 한 형태이다.
     1. `Measurement Update Step`:  
     - $P(X|Z) \propto P(Z|X)P(X)$
       - P(Z|X): Important Weight로 파티클의 `observation`과 `landmark` 사이의 매칭 확률이다. 
       - P(X): Particle로서 각각의 모든 파티클에 대해 Important Weight를 곱하고 큰 값을 기준으로 Resampling 하고 있으니 보정이 되고 있는 것이다.
       - P(X|Z): Posterior
     2. `Motion Predict Step`:
     - $P(X') = \Sigma P(X'|X)P(X)$
       - P(X): Particle 
       - P(X'|X)는 각각의 입자에 대한 이동 모델이고
       - 이것을 다 나타낸 것이 새로운 Particle 인 것이다.

     3. 정리하면, Particle Filter도 Bayisan Filter의 표현형 중 하나인 것이다. 

     3. Original Bayisan Form   ![image](https://sungwookle.github.io/assets/bayes_process.png)


## 3. Implementation of Particle Filter

- 아래의 초록색 박스가 파티클 필터의 `process`
![process_PF](https://video.udacity-data.com/topher/2017/August/5989f54e_02-l-pseudocode.00-00-47-13.still006/02-l-pseudocode.00-00-47-13.still006.png)
- pseudo code로는 아래와 같다.
![pseudo](https://video.udacity-data.com/topher/2017/August/5989f70c_02-l-pseudocode.00-00-16-01.still002/02-l-pseudocode.00-00-16-01.still002.png)
  1. 샘플을 initialize 한다.
  3. 샘플들을 주어진 input에 따라 움직이게 한다.
  4. 샘플들의 `observation`정보와 `landmark`까지의 거리 matching 확률을 계산한다.
  7. ~weight 값을 기준으로 resampling 한다.

- 실제로 파티클 필터를 구현하려고 하면, 센서 데이터를 파티클 필터 기준으로 `TRANSFORM`해야할 것이고, 그 다음 여러 개의 센서를 쓴다면 `ASSOCIATE`해주고 `LANDMARK`와 결합를 지어주어야 할 것이다. 그 다음에서야 matching 확률을 계산해줄 수 있다.
![transformation](/assets/plane_transformation.png)


## 끝