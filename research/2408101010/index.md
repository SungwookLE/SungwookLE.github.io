---
layout: post
type: research
date: 2024-08-10 10:10
category: AI
title: Transformer
subtitle: Transformer Review
writer: 100
post-header: true
header-img: ./img/2024-08-10-21-38-58.png
hash-tag: [AI, Transformer]
use_math: true
toc : true
---

# Transformer Review
> Writer: SungwookLE    
> DATE: '24.08/10    

## 01. RNN
- 연속적인 데이터 (자연어 등)
- 단어 -> 숫자로 바꾼 뒤, MLP에 하나씩 순차적으로 넣어보자

- RNN 동작 방식(Recurrent Neural Network)
    - ![](img/2024-08-10-21-48-34.png)
    - ![](img/2024-08-10-21-57-19.png)
    - $h_t = tanh(x_tW_x + h_{t-1}W+b)$
        - $h_1 = tanh(x_1W_x + b)$
        - $h_2 = tanh(x_2W_x + h_1W_h+b)$
        - ![](img/2024-08-10-21-54-29.png)
        - $\hat{y_3} = h_3W_y + b_y$

- 이렇게 함으로써 얻는 효과는?
    - h가 이전 정보를 담는 역할을 함 (RNN은 이전 정보를 담아낸다)
    - 가변적인 입력을 처리할 수 있다.
        - `나는 OO 입니다` 라는 문장의 토큰이 3개이든, `나는 OO 옷을 입을 OO입니다.` 라는 문장의 토큰이 5개이든, 토큰의 길이가 달려져도 RNN의 파라미터 $W_x, b, W_h$ 학습이 수월하다.
        - 반면, RNN이 아닌 MLP 였다면 토큰의 길이가 통일되어야지만 동일한 네트워크를 사용할 수 있었겠음
        - RNN의 이런 구조 때문에, 아래의 특징이 있음(RNN의 구조적 한계)
        - `next token prediction: GPT`
            1. 멀수록 잊혀진다(`back propagation`) 
            2. 갈수록 흐려진다(`forward propagation`): tanh 때문
            - `vanishing gradient`
    - $L_{total} = L_1 + L_2 + L_3 + L_4 + ...$
        - $\frac{\partial L_4}{\partial W} = weight_1 X_1 + weight_2 X_2 + weight_3 X_3 + weight_4 X_4 + ... $
        - $weight_1 << weight_2 << weight_3 << weight_4 $ 라는 특징이 있다는 의미임
 

- 멀수록 잊혀지는 특징은, 직전 입력에만 큰 학습이 이뤄지게 된다.
    - 멀리있는 단어는 활성화함수의 미분이 중첩(`곱`)되면서 매우 작은 값이 되어지기 때문

- 갈수록 잊혀지는 특징은, forward inferencing에 있어서, `tanh`에 의해서 곱해지는 과정에서 1보다 작은값이 누적곱되면서 작아지기 때문임


- 유형
    1. One to Many : 대표적으로 image captioning
    2. Many to One : 문장의 Score, 감정 점수 등 
    3. Many to Many : 번역기 등 (seq2seq)
        - seq2seq
            - 인코더와 디코더를 각각의 rnn을 붙여놓은것
            ![](img/2024-08-10-22-37-37.png)
            - decoder는 `next token predictor`임
            - encoder의 마지막 h(context vector)를 decoder의 처음h로 사용
            - 53''


