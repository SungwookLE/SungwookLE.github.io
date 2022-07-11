---
layout: post
type: about
title: Resume
header-img: ./img/wall2.jpg
post-header: true
use_math: true
---

# Sungwook LEE
> Email: joker1251@naver.com  
> Github: github.com/SungwookLE  

---------

## 1. EDUCATION

- 한양대학교 자동차전자제어공학과 석사 (4.41/4.5)
    - Intelligent Machine Lab (지도교수: 박장현 교수)    
    - 전공: 시스템 제어 및 추정 로직 SW 
- 국민대학교 기계시스템공학과 학사 (4.04/4.5)

## 2. RESEARCH INTEREST

- Autonomous Driving SW
    - Vehicle Control
    - State Estimation & Prediction
    - Localization

- Deep Learning with Camera Vision
    - Optimized Fusion Algorithm: Inferenced state under Deep Learning + Estimated state based on Dynamic Model

## 3. CAREER

- Researcher(17.1~19.1): `Intelligent Machine Lab, @Hanyang University`
    - RWS + ESC 통합 샤시제어시스템 개발 (17.12~18.06, 현대차 산학과제)
        - 차량 동역학 모델 기반 ESC(제동) + RWS(조향) 통합 최적 제어 로직 개발
        - 모델 상태변수 추정(슬립각, 타이어포스) 로직 개발
        - Actuator Failure 따른 고장 허용 제어
    - ADAS 주요센서의 고장 및 대응 전략 기술 개발 (17.04~17.12, 정부 국책과제)
        - Camera + Lidar + Radar Sensor Fusion SW 알고리즘 개발 및 실차 테스트
    - 기계학습을 이용한 타이어 저압진단 AI 알고리즘 개발 (17.06~17.12, 연구과제)
        - 휠속의 이상 감지를 통한 저압진단 AI 알고리즘 개발

- Researcher(19.1~): `Safety Control, @Hyundai Motor Company`
    - 차량 상태 추정 SW 개발: 전복 감지
        - 차량 롤각 추정 SW 로직 개발
        - 주행 중 롤각 추정 성능 개선
        - 6DOF 각가속도 오프셋 제거 로직 개선
    - 승객 영상 인식 자세 추정 SW 개발
        - Deep-learning 기반 승객 keypoints 추정 및 3D 좌표 추정
        - Occlusion 강건화 알고리즘 개발
        - 실내 승객 클래스 판단 알고리즘 개발
        - 승객 영상 SW 아키텍쳐 분석 및 요구사항 설계
    - 에어백 SW 검증
        - 요구사항 기반 테스트 케이스 자동화 생성 코드 개발
        - 정적/동적 SW 검증, SILS/HILS 기능 검증
        - 에어백 SW 디자인리뷰 담당
    - 충돌시험 빅데이터 분석툴 개발
        - Django, 충돌데이터 API 개발
        - 충돌데이터 입출력 및 전처리 자동화 기능 개발
        - 기계학습을 이용한 충돌 데이터 분석 기능 개발 

## 4. SKILLS

- Model Based Development: Control & Estimation
    - Advanced Control, Optimal Estimation
    - **Nonlinear & Linear Kalman Filter**
- **Programming Languages: C++, C, Python**
- Framework: Tensorflow, Keras, Django
- Tool: Git, Matlab, Simulink, MySQL, Docker, AWS

## 5. PAPERS, PATENTS, CERTIFICATE

- PAPERS(*1st author)
    - "Control Allocation of Rear Wheel Steering and Electronic Stability Control with Actuator Failure", *IEEE International Conference on Vehicular Electronics and Safety 2018*
    - "Integrated Chassis Control of Suspension and Steering Systems for LKAS", *International Conference on Control, Automation and Systems 2017*

- PATENTS(*1st author)
    - RWS, ESC Actuator 고장허용제어 기술 특허 출원 
    - 6DOF IMU 센서 오프셋 가속 제거 기술 특허 출원
    - 기계학습 기반 충돌 승객 상해 예측 안전 제어 기술 특허 출원
    - Kinetics 모델 기반 실내 승객 거동 추정 기술 특허 출원
    - 실내 영상 기반 승객 거동 추정 및 안전 제어 기술 특허 출원

- CERTIFICATE
    - 국가기술자격: 빅데이터 분석기사 (취득일: 22.7/15)

## 6. PROJECT EXPERIENCE
### 1. Integrated Chassis Control (17.6~18.6)
- Vehicle Dynamics Model Based, Optimal Control and Estimation SW algorithm(`Matlab/Simulink`)
    - Rear Wheel Steering and Electronic Stability Contorl
    - Controller: Nonlinear Sliding Mode Control
    - Estimator: Tire Force and Side Slip Angle with Dual Extended Kalman Filter (모델 파라미터 및 상태변수 동시 추정)
    - Optimizer: KKT(Karush-Kuhn-Tucker) method for considering Contraints

        ||
        |-|
        |![](./img/thesis.png)|

### 2. 차량 상태 추정 SW 개발: 롤각 추정 (20.4~20.10)
- Dynamics Model Based, Roll Angle Estimation SW algorithm(`c,c++`)
    - 3차원 거동 상황에서의 강건한 롤각 추정 로직 개발
    - 차량 신호 분석 기반 Gain Scheduling Domain Design
    - Gain-Scheduled Roll Angle Estimator 개발
    - 롤각 추정 Fault-Tolerant 로직 개발
    - 5DOF IMU 신호 오프셋 제거 (FOC, SOC) 개선 로직 개발

### 3. 승객 영상 인식 SW 개발 (21.4~21.11)
- Deep Learning algorithm
    - Deep-learning 기반 승객 2D keypoints 추정 및 3D 자세 추정(`pytorch, opencv`)
    - Occlusion 강건화 알고리즘 개발

- 승객 클래스 판단 알고리즘 개발(`keras, tensorflow`)
    - autoencoder 활용한 semi-supervised learning
    - xAI(explainable AI) 분석 기능 개발

        |||
        |-|-|
        |![](./img/safety_pose.gif)|![](./img/safety_class.gif)|

- 승객 영상 SW 아키텍쳐 분석 및 설계 (22.6~)

## 7. EDUCATION
### 1. Self-Driving Car Nanodegree, Udacity (20.10~21.2)
- Computer Vision
- Sensor Fusion
    - Pedestrian {위치 , 속도} 예측 Lidar+Radar Sensor Fusion
    - Kalman Filter, Extended Kalman Filter
    - [Lidar+Radar Sensor Fusion with Extended Kalman Filter](https://github.com/SungwookLE/udacity_extended_kf)

        ||
        |-|
        |![](./img/lidar_radar_fusion.png)|

- Localization
    - Particle Filter

- Planning, Control
- System Integration: `ROS`

### 2. C++ Intermediate Nanodegree, Udacity (20.6~20.9)
- Object-Oriented Programming
    - A-Star Search Algorithm
- Memory Management
- Concurrency Programming

### 3. Computer Vision Nanodegree, Udacity (22.6~22.9)
- Advanced Computer Vision and Deep Learning
- Object Tracking Localization
    - SLAM

## End