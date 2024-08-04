---
layout: post
type: research
date: 2023-05-08 21:15
category: AI
title: LaneGCN
subtitle: Predict Target Path
writer: 100
post-header: true  
header-img: ./img/2023-05-08-21-16-29.png
hash-tag: [PathPrediction, Argoverse]
use_math: true
toc : true
---

# Paper Review: LaneGCN 
> Writer: SungwookLE    
> DATE: '23.05/08    
> [Repo](https://github.com/uber-research/LaneGCN), [Slide](http://www.cs.toronto.edu/~byang/slides/LaneGCN.pdf), [Oral](https://yun.sfo2.digitaloceanspaces.com/public/lanegcn/video.mp4), [Paper](https://arxiv.org/abs/2007.13732)  


## Abstract
1. Instead of encoding vectorized maps as raster iamges, we construct a lane graph
2. LaneGCN which extends graph convolutions with multiple adjacency matrices and alone-lane dilation
    - [graphConvNet](https://github.com/heartcored98/Standalone-DeepLearning/blob/master/Lec9/Lec9-A.pdf)

3. 현재의 Scene 정보로만 예측을 하는 것이 아닌, 과거의 History 정보를 Regression 하여 Lane 관련한 경로 예측을 수행 
    - 예측만을 목적으로 한다면, AI가 아니어도 안정적으로 접목할 수 있는 방법이 있지 않을까?
