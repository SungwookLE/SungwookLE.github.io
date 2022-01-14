---
layout: post
type: research
date: 2022-01-14 10:10
category: Kaggle
title: CreateArt using GAN
subtitle: Use GANs to create art - will you be the next Monet?
writer: 100
post-header: true
header-img: img/horse2zebra.gif
hash-tag: [PyQt5, backend, database, db, XingAPI, Trade, Stock]
use_math: true
---

- toc
{:toc}

# I'm Something of a Painter Myself
> Author: [SungwookLE](joker1251@naver.com)  
> Date  : '22.01/14  
> Kaggle Competition: [LINK HERE](https://www.kaggle.com/c/gan-getting-started/overview)  
> Reference: [#1](https://www.kaggle.com/amyjang/monet-cyclegan-tutorial), [#2](https://www.kaggle.com/dimitreoliveira/introduction-to-cyclegan-monet-paintings)
> Paper: [arxiv_paper](https://arxiv.org/pdf/1703.10593.pdf  
> Repository: [git-repo](https://github.com/junyanz/CycleGAN)


## 1. Overview & Challenge

- Computer vision has advanced tremendously in recent years and GANs are now capable of mimicking(흉내내다) objects in a very convincing(설득력있는) way. 
- But creating museum-worthy masterpieces is thought of to be, well, more art than science.
- So can (data) science, in the form of GANs, trick classifiers into believing you’ve created a true Monet? That’s the **challenge** 

### 1-1. Challenge
- A GAN consists of at least two neural networks: a generator model and a discriminator model. 
    - The generator is a neural network that creates the images. 
- For our competition, you should generate images in the style of Monet. This generator is trained using a discriminator.
- The two models will work against each other, with the generator trying to trick the discriminator, and the discriminator trying to accurately classify the real vs. generated images.
- Your task is to build a GAN that generates 7,000 to 10,000 Monet-style images.

### 1-2. Data description

- The monet directories contain Monet paintings. Use these images to train your model.
- The photo directories contain photos. Add Monet-style to these images and submit your generated jpeg images as a zip file. 
- Files
    - monet_jpg - 300 Monet paintings sized 256x256 in JPEG format
    - monet_tfrec - 300 Monet paintings sized 256x256 in TFRecord format
    - photo_jpg - 7028 photos sized 256x256 in JPEG format
    - photo_tfrec - 7028 photos sized 256x256 in TFRecord format
- Submission format
    - Your kernel's output must be called images.zip and contain 7,000-10,000 images sized 256x256.


## 2. 노트북 
- [#2](https://www.kaggle.com/dimitreoliveira/introduction-to-cyclegan-monet-paintings)
- virtual-env 구성이 잘 안되어서 노트북 실행을 못하였다.
- 다시 시도해보자.. (1/14)

<center><img src='https://raw.githubusercontent.com/dimitreOliveira/MachineLearning/master/Kaggle/I%E2%80%99m%20Something%20of%20a%20Painter%20Myself/banner.png' height=350></center>




## 끝