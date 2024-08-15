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
draft: false
---

# Transformer Review
> Writer: SungwookLE    
> DATE: '24.08/10    
> 출처: 혁펜하임님의 강의 TTT(Transformer)    

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
            - cell은 plain RNN보다는 LSTM이나 GRU를 주로 사용
                ![](img/2024-08-11-13-35-14.png)
                - 그러나, 멀수록 잊혀지는 문제는 해결되지 못함
                - LSTM은 인풋(X)와 히든(h)에 대한 밸브(0~1)를 두고 학습시킴으로써 문제를 해결하려고 했으나, 여전히 멀리 위치할 수록 학습 시에 누적(곱)에 의해 영향을 덜 받게 됨
        - seq2seq 구조의 문제점
            1. 멀수록 잊혀진다. (인코더, 디코더 모두에 해당)
            2. context vector에 마지막 단어의 정보가 가장 뚜렷하게 담기니, 그런 h로 decoder가 번역하다 보니 마지막 단어만 제일 열심히 본다. (인간의 사고 방식이 아님)
                - 트랜스포머는 'Attention`을 이용한 것으로, 어떤 단어에 집중할지를 선택할 수 있게 된다.

- Insight for Attention
    - `RNN+Attention`
        - 인식한 문제점 
            1. seq2seq는 왜 마지막 context vector만을 디코더에 전달하느냐,
            2. 입력된 단어 토큰 중, 어떤것을 더 집중해야할지를 반영할 수 있어야 하는데 말이지.
        - 개선 
            ![](img/2024-08-11-15-21-17.png)
            1. 디코더에서 사용하는 context vector를 전달하는 부분에서 Attention mechanism을 이용하자!
            2. seq2seq에서 $\hat{y_4} = s_4W+y + x_4W_x$였다면, $\hat{y_4} = s_4W_y + c_4W_y'+x_4W_x$로 바꾼 뒤 $c_4$를 attention으로 만들어보자
                - 내적과 가중합을 이용
                - $c_4 = <s_4,h_1>h_1 + <s_4, h_2>h_2 + <s_4, h_3>h_3$로 구함으로써, 어떤 임베딩 벡터 $h$에 주목하여야할지를 학습하자.
        - 한계
            1. 문제는 여전히 h와 s를 RNN의 chain을 이용하기 때문에 발생하는 멀수록/갈수록 잊혀지는 문제
            2. 디코더에서만 attention된 context vector를 사용하는 것이기 때문에 `멀수록 잊혀지는 문제`는 여전함
            3. 심지어, 시점상 뒤에 있는 단어는 참고조차 하지 않는 문제가 있음 (Chain으로 이어져 있으니까)
            - 즉, 의미를 제대로 못 담은 h에 attention 한다.

    - 트랜스포머는 위의 문제를 해결하기 위해, chain을 다 끊어버림
        - chain을 끊었다는 의미가, RNN을 없애고 self-attention을 사용했다는 의미임
        - [트랜스포머는 attention을 적극 활용](./img/필기1.jpg)
            - RNN을 완전히 버렸다.
            1. Decoder가 마지막 단어만 열심히 보는 문제(갈수록 흐려진다)를 attention으로 해결
            2. 학습 시, 멀수록 잊혀지는 문제를 self-attention으로 해결 
            3. 의미를 제대로 못 담은 h에 attention (갈수록 흐려진다)를 self-attention으로 해결
        - `Self Attention` 
            - c: context vector (attention, 인코더-디코더 연결)
                - $c_4 = <s_4, h_1>h_1 + <s_4, h_2>h_2 + <s_4, h_3>h_3$
            - s: sequence vector (self-attention, decoder, `masked self attention`)
                - $s_4 = <s_4, s_1>s_1 + <s_4, s_2>s_2 + <s_4, s_3>s_3 + <s_4, s_4>s_4 $
                - $s_5$는 정답이기 때문에 알려주어선 안되지
                    - ~~+<s_5, s_4>s_5~~
            - h: word embedding vector (self-attention, encoder)
                - $h_2 = <h_2, h_1>h_1 + <h_2, h_2>h_2 + <h_2, h_3>h_3$
            ![](img/2024-08-11-15-26-44.png)
            - 참고: 위의 수식에서 h를 생성하는데 사용하는 x는 (단어와 위치 정보)가 임베딩된 상태여야 한다.
                - chain을 끊어냈기 때문에, 순서정보를 담아서 전달해 주어야함

## 02. Transformer - Attention is all you need

- 자연어 언어 모델에서 강력한 성능
- 이미지 분야에서 CNN이 있다면, 언어 분야에는 트랜스포머(self-attention)가 있는 것
- self-attention을 활용한 모델로, 내적을 이용한 weighted sum을 이용하여 단어 간의 관계를 효과적으로 학습하고 표현함
![](img/2024-08-11-15-36-58.png)

1. Transformer
    1. Embedding 구조
        - input이 이미지라면 (`개수x채널x사이즈hx사이즈w`)
            - 개/채/행/열
        - input이 문장이라면 (`문장 개수x가장 긴 문장의 단어 개수x임베딩된 단어의 차원 크기`)
            - 개/단/차
            - 문장마다 단어 개수가 다르니, 이를 맞춰주기 위해 <pad> 토큰을 이용해서 길이를 맞춰준다.

        1. Input Embedding
            - one-hot 인코딩 되어 있는 것을 FC layer에 통과시키는 것
        2. Positional Encoding 
            - Embedding만 되어서는 순서정보를 담을 수 없다.
            - 단어의 위치를 one-hot 해서, 그것을 FC layer에 통과시키는 것
            - 순서 정보를 알려주어야 한다.
            - `트랜스포머 논문 자체`에서는 Positional Encoding 자체의 FC는 학습시키지 않고 고정된 벡터를 사용함
                - `sin, cos` 함수 사용함
            - 이후 논문에서는 여기에서의 파라미터도 학습의 대상으로 함
        3. Input Embedding + Positional Encoding 두개를 더하여 Embedding 정보를 만든다.

    2. Encoder 전체 구조
        - Multi-Head Attention
            - ![](img/2024-08-11-16-24-41.png)
            - Key, Query, Value
                - ![](img/2024-08-11-16-39-08.png)
            - Query: 관계를 물어볼 기준 단어 벡터 
                - `질문`
            - Key: Query와 관계를 알아볼 단어 벡터
                - `답변`
            - Value: 키 단어의 의미를 담은 벡터
                - `표현`
            - 키,쿼리,밸류의 역할은 각각 다르기 때문에 이를 각각의 FC 레이어로 학습함
            - 이러한 키,쿼리,밸류 set을 CNN의 필터 개수처럼 여러개 배치한 것을 `Multi-Head Attention`이라 부름
                - 멀티 헤드에서 똑같은 Loss를 가지고 Back-propagation 하더라도, 파라미터 초기값이 서로 다르니, 각각 다르게 바이어스를 가지게끔 학습됨
                - 멀티 헤드를 둠으로써, 여러 필터를 두는 것과 같은 효과가 발생한다. (~=CNN의 복수개의 필터)
            - `Multi-Head Attention`의 효과
                - 멀티 헤드의 하나의 헤드 헤드가 각각 바이어스를 가지고 학습되니, 앙상블의 효과, `집단지성`과 유사한 효과가 나타남
            - Scaled Dot-Product Attention 이후 Concat->Linear를 통과하게 되면서 여러 헤드의 출력이 조화가 이루어지게 됨
                - `의견교류`
            - 이 전체를 N번 반복하여 인코더에서 `word embedding vector(h)`를 만들어냄
        
    3. Decoder 구조
        - `Masked Multi-Head Attention`
        - 학습 시에는 `next token`의 참값을 집어넣어서 지도학습하고, test 땐 자신의 출력을 입력으로 사용함
        - `Masked`를 해주어야 하는 이유는 정답을 보여주고 학습시킬 순 없으니까..!
    4. Encoder-Decoder Attention 구조
        - ![](img/2024-08-13-23-48-22.png)
        - Q로는 디코더 레이어에서의 출력 임베딩 벡터를, K/V는 인코더 레이어에서의 출력 임베딩 벡터를 사용함

        - 3번+4번 `self-atten -> enc-dec-atten -> FF` 순서로 통과하게됨
            - 디코더로 문장을 파악 -> 입력 문장에서 뭘 주목하는지 보고 -> 다음 단어 예측
        
    - softmax(내적+가중합)를 수행하는 attention 함수
        - 이를 통해 어떤 것에 주목하여야하는지를 학습할 수 있게 됨
        - ![](img/2024-08-13-23-59-02.png)
        - ![](img/2024-08-15-15-24-05.png)

    - 단어 사이 사이에 어떤 단어 끼리 관계가 있는지를 학습하는 것이 Attention 임
        - ![](img/2024-08-14-00-03-45.png)
        - 왜 관련 없는 단어이 벡터가 Query에 수직이 아닐까?
            - 수직이면 cos 내적은 0이 될텐데
            - 소프트맥스를 취하였기 때문임
                - 내적이 0이 나오는거 보다 음수로 나오는게 더 작은 값이 되니까

    - 추론할때는 단어 SET을 넣어주어야함
        - `next token predictor`이기 때문에 알고자하는 단어 이전에 등장한 모든 단어를 입력해주어야함

2. Transformer Evaluation
    - 평가를 위한 지수임
    - 문장은 주관적이니까, 이미지 분류 문제처럼 1 또는, 0으로 평가할 수는 없음
    - `PPL, BLEU score`를 주로 사용함
    
    1. PPL(Perplexity)
        - 당혹감, 헷갈려하는 정도를 평가하기 위한 것
        - CrossEntropy에 exp를 취하면 PPL이 됨
        - 문장 전체에 대해서 헷갈려하는 정도에 대한 metric임

    2. BLEU score
        - 번역task에 있어선 많이 쓰이는 편
        - N-gram precision을 이용: 연속한 n 개 단어가 정답 문장에 존재하는지
        - Unigram이 아니라, N-gram을 쓰는 이유는 순서까지도 맞추어야하기 때문임

    - 0h 35'~ 부터

## 03.

