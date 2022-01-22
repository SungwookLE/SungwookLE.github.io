---
layout: post
type: algorithm
date: 2021-12-05 10:10
category: 코딩테스트 연습
title: N으로 표현 Lv3
subtitle: 프로그래머스 동적계획법
writer: 100
hash-tag: [Dynamic_Programming, Programmers]
use_math: true
---

# 프로그래머스 > 동적계획법 > N으로 표현
> AUTHOR: SungwookLE    
> DATE: '21.12/05  
>> PROBLEM: [문제링크](https://programmers.co.kr/learn/courses/30/lessons/42895)  
>> REFERENCE: [참고](https://velog.io/@euneun/%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%A8%B8%EC%8A%A4-N%EC%9C%BC%EB%A1%9C-%ED%91%9C%ED%98%84-DP-%EB%8F%99%EC%A0%81%EA%B3%84%ED%9A%8D%EB%B2%95-C)  
>> LEVEL: Lv3    

## 1. 혼자선 못 품
- `DP`로 풀라고해서, 처음에는 `string`으로 모든 식을 표현한 다음에 `string calculator`를 만들어서 계산을 하고, 계산한 값이 `number`와 맞는지 비교해서 문제를 풀려고 했다.
- 당연히 비효율적이고, `string calculator`를 만드는 것도 어려워서 후위계산법 등을 찾아봐야만 했다.

- 본 문제는 **Memoization**을 활용해서 풀어야하는 문제로 메모이제이션은 컴퓨터 프로그램이 동일한 계산을 반복해야 할 때, 이전에 계산한 값을 메모리에 저장함으로써 동일한 계산의 반복 수행을 제거하여 프로그램 실행 속도를 빠르게 하는 기술이다. 동적 계획법의 핵심이 되는 기술이다. 메모아이제이션이라고도 한다. 

- 동적 계획법을 이용한 문제풀이는 다음의 *룰(Rule)* 을 기억하면 좋다고 한다.
- ![image](./img/1.png)

- 그리디 문제도 어렵고, DP문제도 넘나 어려운 것.. ㅠㅠ

## 2. 다른 사람 풀이 참고
> 5가 두번 이용된 5/5의 경우 5가 한번 이용된 경우를 사칙연산으로 결합한 결과임을 알 수 있다.  

- 여기서 N을 i번 이용했을때 만들 수 있는 수들을 DP[i]에 저장하면 될것이라는 생각을 할 수 있다.
- 즉 DP[i] : i개의 N으로 만들 수 있는 숫자들 이다

- 실제로 dp배열에 저장해보자. 유의할 것은 DP 배열의 인덱스값은 0부터 시작하므로 실제 이용되는 값보다 1만큼 작다는것! 그리고 아래에서 ㅁ은 사칙연산을 의미한다!

    - DP[0] : 1개의 N으로 만들 수 있는 수들의 집합은 N한개 밖에 없다.
        - {N} : N1이라고 하자
    - DP[1] : 2개의 N으로 만들 수 있는 수들의 집합은 NN과 N1(N 한개로 만들수있는수)두개끼리 사칙연산한 결과로 이루어져있을것이다.
        - {NN, N1ㅁN1} : N2라고 하자.
    - DP[2] : 3개의 N으로 만들 수 있는 수들의 집합은 NNN과 N1(N 한개로 만들수있는수)와 N2(N 두개로 만들수있는수)를 사칙연산한 결과로 이루어져있을 것이다.
        - {NNN, N1ㅁN2, N2ㅁN1} : N3라고 하자.
    - DP[3] : 4개의 N으로 만들 수 있는 수들의 집합
        - {NNNN, N1ㅁN3, N2ㅁN2, N3ㅁN1} : N4

## 3. unordered_set의 간단한 사용법
- 간단한 사용법
    - set에 삽입할때는 insert라는 함수를 사용하고 find라는 함수를 사용하여 해당원소가 set에 있는지 확인 가능하다.
    - 이때 해당원소가 없으면 set.end()를 반환한다.

## 3. 코드
- 코드를 써보면서 반복적으로 보면 좋을 것 같다.

```c++
#include <vector>
#include <unordered_set>

using namespace std;

int get_Ns(int N, int idx){
    // NN(idx ==1), NNN(idx==2), NNN(idx==3)... 과 같은 형태 만드는 함수
    int result = N;
    for(int i = 1 ; i <= idx ; ++i){
        result = result * 10 + N;
    }

    return result;
}

int solution(int N, int number){
    if (N==number) return 1; //N과 number가 같다면, N을 한번 사용해서 number를 만들 수 있음

    vector<unordered_set<int>> DP(8);
    // DP에 저장할 것 -> DP[i] : 1개의 N으로 만들 수 있는 숫자들

    DP[0].insert(N); //한개의 N으로 만들 수 있는 수는 N뿐임

    for(int k = 1 ; k < 8 ; k++){

        // DP[k]에 NNN ... (k+1만큼 반복)과 같은 형태 삽입
        DP[k].insert(get_Ns(N,k));

        // DP[k]에 사칙 연산의 결과또한 삽입
        for(int i = 0 ; i < k ; ++i){
            for(int j =0; j < k ; ++j){

                if(i+j+1 != k) continue;
                
                for (int a : DP[i]) {
                    for(int b : DP[j]){
                        DP[k].insert(a+b);
                        // 검사가 필요한 연산들

                        // (1) 음수 존재하면 안됨
                        if ( a-b> 0)
                            DP[k].insert(a-b);
                        
                        DP[k].insert(a*b);

                        // (2) 0 존재하면 안됨
                        if ( a/b> 0) DP[k].insert(a/b);

                    }
                }
            }
        }

        if (DP[k].find(number) != DP[k].end()) //DP set에 number에 해당하는 값이 있으면 k+1을 반환
            return k+1;

    }

    return -1;
}
```

## 끝