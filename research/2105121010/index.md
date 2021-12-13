---
layout: post
type: research
date: 2021-05-12 10:10
category: PLAN
title: Study Plan
subtitle: 어떻게 하는게 좋을까, 주기적 업데이트 중
writer: 100
post-header: true
header-img: 153008571329334.jpeg
hash-tag: [plan]
use_math: true
---

- toc
{:toc}

## 1. Study Plan @2021.5
  > Date: '21.5/12
  >> 최근, 뭔가 루즈해졌다, 그게 나쁜건 아닌데 이런 순간이 오면 조만간 슬럼프가 찾아올 수 있기 때문에 정리가 필요한듯싶다,
  어디까지를 탐구하고, 어디까지를 내 것으로 만들어야 할지, 좀 더 쉽고 간결한 미션

### A. 세팅
  1. 원래 하던 것에 있어선, 잘해야,,
      - SIMULINK/C기반 위치추정/제어 로직, 동역학/제어이론
      - C++기반 위치추정(localization), 확률기반 필터
      - 칼만 필터, 최적화기법 등

  2. 새로운 분야에 있어선, 인사이트 ex: 프로젝트 참가 등
      - 카메라 기반 opencv 알고리즘 
      - python기반 딥러닝 tensorflow
      - ROS <- 로직개발에 좀 더 편리한 나에게 유용

  3. SW 이슈/관리 능력
      - git, SVN, conda, docker
      - vcdm, jira,, testing/realese
      - 아키텍트 연습/고민, (품질있는 SW가 경쟁력)

  4. 출입권: 코딩 테스트(알고리즘)
     - 이건 문제 꾸준히 많이 푸는게,

### B. 끝으로..

  ```c++
  int main(){
    std::cout << "To infinity" << std::endl;
    return 0;
  }
  ```
---

## 2. Study Plan @2021.11
> Date: '21.11/15  
> Author: SungwookLE
>> 불안하지... 걱정만 하면, 더 빠져든다.  
>> 현실에 최적화된 계획으로 준비하라

### A. 공부하면 좋겠다고 떠오른 것
- 학습: [인프런](https://www.inflearn.com/) 등?
    - 알고리즘(코딩) 테스트 스터디 (기초부터)
    - 트레이딩 시스템 개발(LSTM 등 기계학습 기반 이용): 차트 분석 및 오토 트레이딩 봇 만들기
- Daily 커밋과 컨디션 관리에는 코테 문제풀기가 매우 좋다.

### B. 직업적 전공 영역
- 학습(필요시): [Udacity](https://www.udacity.com/)?
    - 딥러닝 기반 Human Pose: Vision 기반의 딥러닝 모델 개발 [v]
    - 모델 기반 상태/파라미터 추정기: 모델 기반의 상태 추정 모델 개발 [v]
    - Fusion Algorithm: Control/Bayes/Optimization 기반의 칼만필터 등을 이용한 Inference Fusion [v]

### C. 프로그래밍 역량
- 무조건, 헤게모니 전환 시점은 온다
    - C++
    - python
    - ROS

### D. 끝으로..
내일(11/16) 부터 워밍업..
코딩테스트 문제 푸는 것으로 시작하자
(하루 3시간 Rule, 하기 싫은 날은 카페가서)

## 3. Study Plan @2021.12
> Date: '21.12/13  
> Author: SungwookLE
>> 유다시티..Nanodegree.. 매번 프로모션이 있기는 한데, ~12/19 까지 75% OFF 이다.

### A. [Robotics](https://www.udacity.com/course/robotics-software-engineer--nd209)
- 가격: 4개월 간 406,769원
- [SYLLABUS](./img/nd209_Robo_syllabus_v2.pdf)
  - 기계 제어/추정 SW를 실제적으로 HW와 결합하여 제품으로 끌어내기 위해선, `ROS`라는 플랫폼을 사용하여야 될 것으로 판단하에,,,
  - 지금까지 공부한 것을 하나의 시스템(SW+HW)으로 끌어내보려고 강의에 관심이 있음

### B. [AI for Trading](https://www.udacity.com/course/ai-for-trading--nd880)
- 가격: 6개월 간 610,153원
- [SYLLABUS](./img/AI+for+Trading+Learning+Nanodegree+Program+Syllabus.pdf)
  - 실라버스 안에 내용은, `torch, scikit-learn` 등을 이용하여 `common`한 pipeline을 따라 NLP도 하고, 차트 분석도 하는 것으로 보임
  - Kaggle의 문제 [예제](https://www.kaggle.com/c/two-sigma-financial-modeling/overview/description) 하나 해보면 좋을 것 같음(12/14)
    - 사람들 코드도 살펴 보고, 직접 해보면서, 강의를 들을 필요가 있을지 생각해 보자.

### C. [Flying Car and Autonomous Flight](https://www.udacity.com/course/flying-car-nanodegree--nd787)
- 가격: 4개월 간 406,769원
- [SYLLABUS](./img/Flying+Car+Nanodegree+Syllabus.pdf)

### D. [Computer Vision](https://www.udacity.com/course/computer-vision-nanodegree--nd891)
- 가격: 3개월 간 305,077원
- [SYLLABUS](./img/Computer+Vision+Nanodegree+Syllabus.pdf)
  - `Human Pose`에 도움이 될까 하여 본 것인데, `Object Detecting` 쪽에 좀 더 맞춰져 있고, `SLAM` 프로젝트가 있어 흥미가 갔는데, `SLAM`은 로보틱스 강의에서 좀 더 자세하게 다룬다. 
    - 그래서, 후순위로 뺐음


## 끝
- it ain't about how hard you hit, it is about how hard you can get hit and keep moving forward, how much can you take and keep moving forward  
![image](img/153008571329334.jpeg)
