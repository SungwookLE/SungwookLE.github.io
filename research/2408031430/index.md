---
layout: post
type: research
date: 2024-08-03 14:30
category: AI
title: Vectorized Scene Representation for Efficient Autonomous Driving
subtitle: End to End
writer: 100
post-header: true
header-img: ./img/2024-08-03-14-31-33.png
hash-tag: [E2E, Autonomous]
use_math: true
toc : true
---

# Paper Review: Vectorized Scene Representation for Efficient Autonomous Driving
> Writer: SungwookLE    
> DATE: '24.08/03    
> [Paper](https://arxiv.org/pdf/2303.12077), [Git](https://github.com/hustvl/VAD)

- 23년 8월에 SOTA 성능을 보여준 E2E Planning 논문임**

## Abstraction
- 자율주행차량의 경로 계획을 위해선 주변 환경에 대해 comprehensive understanding이 필요함
- 이를 위해선 Raseterized Scene이 아닌 Vectorized Scene이 필요하다.
![](img/2024-08-03-15-40-04.png)

- 본 논문은 end-to-end fully Vectorized 패러다임을 제시함
    1. Agent의 움직임과 맵 정보를 vectorized 해서, 경로 계획에 더 효과적임
    2. 더 빠르다. better than sparsed rasterized information

## 1. Introduction
- Traditional autonomous driving 은 모듈러 패러다임으로 설계되었다.
    - 즉, 인지, 예측, 판단, 계획 모듈 들로 이루어져 있었다.
    - 이러한 모듈로 인해 플래닝 모듈은 cannot access the original sensor data (풍부한 의미적 정보를 담고 있는)
    - 플래닝은, 모든 처리가 완료된 정보만을 수신함으로써, 인지 오차 등이 경로 계획에 큰 에러를 발생시켰다. (안전 측면에서 안좋음)

- 최근의 End-to-End 접근은 센서의 로우 데이터를 입력받아 플래닝을 출력하는 통합 디자인 되는 방식으로 진행되었고, 어떤 연구는 학습 방식이 아닌 방법으로 접근되기도 하였으나, 최적화가 매우 어려운 한계가 있다. (학습된 무언가가 필요해)
- 대부분의 End-to-End AI 접근은 rasterized scene을 이용해서 플래닝을 수행하였다.
    - 그러나, 이러한 방식은 메모리측면에서 비효과적이고, 또, 몇몇 중요한 데이터 (예를들면 장애물) 같은 정보 표현에 있어 해상도 문제등으로 누락되는 큰 문제가 있다.
    - vectorized 방식(본 논문에서 제시하는)이 더 우수하다
        1. 교통 흐름, 주행방향(일방통행 등) 까지도 수월하게 표현이 가능해서, 경로 계획을 수행할 때 공간 탐색에 더 유용한 정보를 제공한다.
        2. 연산도 효율적이고..,
- 이를 위해, SW 적으로는 map queries 와 agent queries 를 이용해서 현재 필요한 정보는 interaction 한다.
- 또, VAD(본 논문에서 제시하는 모델의 이름)는 3개의 instance-level planning constraints를 제시하였다.
    1. Ego-agent 충돌 제약조건
    2. Ego-boundary 제약 조건 (도로 경계)
    3. Ego-LaneDirection 제약 조건 (역주행)

## 2. Related Work
- **Perception**
    - Bird's-eye view (BEV) representation has become popular and has greatly contributed to the field of perception.
    - `BEVFormer` 라는 모델은 카메라 only 정보만을 가지고 spartial(공간적) and temporal attention 을 제안했고, BEV feature를 효과적으로 인코딩했다.
    - 본 논문에서는 BEVFormer과 MapTR(hdmap 을 vectorized하는 네트워크임)을 이용해서 Perception 네트워크를 구성했다.

- **Motion Prediction**
    - traditional 한 방식은 perception 정보를 참값으로 가정한 뒤 이를 통해 경로를 예측하는 것임
    - 다른 연구에서는 네트워크를 이용한 방식이 있었는데, 이미지 기반의 네트워크라던지(BEV 이미지), 그래프네트워크가 제안되기도 하였음
    - 최신의 연구는, perception와 prediction을 동시에 수행하는 (jointly) 연구가 있었음
    - [PIP](https://arxiv.org/pdf/2212.02181)라는 네트워크에서는 Vectorized map 정보와 Agent의 인터랙션을 반영해서 경로를 예측하였음 (SOTA)
    - 본 논문에서는 PIP 의 정보를 기반으로 Prediction 정보를 생성함

- **Planning**
    - 최근에 연구들은 perception과 prediction을 생략하고, 바로 planning 또는 제어입력을 만들어내는 연구가 수행 중임.
    - 이런 컨셉은 간단해보이기는 하나, 해석불가능한 단점이있고 또, 최적화도 어렵다.
    - 강화학습은 플래닝에 적합한 방식이며, 많은 연구가 시도되고 잇음.
    - dense한 cost map 기반의 플래닝도 다양히 연구되고 있으며, 최소 cost 지점으로 도달하기 위한 플래닝 방식임
    - 근데, 이런 방식은 hand-craft 하기도 하고, 잘 튜닝되면 성능이 좋으나, 그게 어려움 (후처리는 안하는게 좋아)
    - 본 논문에서는 vectorzied로 표현된 장면을 이용해서 플래닝을 시도해보았다.

## 3. Method

- **Overview**
    - ![](img/2024-08-03-16-50-58.png)
    - 여러 프레임의 멀티 카메라의 이미지를 입력받고, BEV Encoding을 수행함
    - agent(인식된 트랙의 센서 로우값?) 및 map 쿼리를 이용해서, 정보를 입력받은뒤 각각의 vectorized transformer를 이용해서 입력정보를 처리한다.
    - 

