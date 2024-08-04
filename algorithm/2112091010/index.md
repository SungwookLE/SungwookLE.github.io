---
layout: post
type: algorithm
date: 2021-12-09 10:10
category: 코딩테스트 연습
title: 도둑질 Lv4
subtitle: 프로그래머스 동적계획법
writer: 100
hash-tag: [Dynamic_Programming, Programmers]
use_math: true
---


# 프로그래머스 > 동적계획법 > 도둑질
> AUTHOR: SungwookLE    
> DATE: '21.12/09  
>> PROBLEM: [문제링크](https://programmers.co.kr/learn/courses/30/lessons/42897)  
>> REFERENCE: [참고](https://yabmoons.tistory.com/477)  
>> LEVEL: Lv4    

## 1. 혼자 풀지 못함..

- 생각해본 아이디어..
    - 첫집부터 -> 마지막집 까지 돈을 더해나가고, 이 때 특별한 조건인, 인접한 두 집을 터면 안된다는 것을 만족하게..
    - 그렇게 구한 값의 최대값을 리턴, (중간 중간 값 비교를 해서 DP에 담아야 겠네..)

    - 점화식을 세우기가 어려워서,, 감이 안잡혔다.
    - 다른 사람의 코드를 참고해보니,
    - DP `vector`를 2개 만들고 첫번째 집을 턴 경우와 털지 않은 경우를 각각 `starting point`로 하여 DP 문제를 풀었네.


## 2. 다른 사람 풀이 참고함

- x번 집을 털겠다 라는 것은, x - 1번 집과, x + 1번 집은 털지 못한다는 것을 의미한다.
    - 즉 ! x - 2 번을 털었을 때의 최대액수 + x번 집을 털었을 때 훔칠 수 있는 액수가 "x번 집을 터는 경우의 구할 수 있는 최대액수"가 된다. 위에서 소개한 변수로 표현해본다면 DP[x] = DP[x - 2] + money[x] 가 된다.

- 반대로 털지 않는다면 어떻게 될까 ??
    - x번 집을 털지 않는다는 것은 x - 1번 집을 털 수 있다 라는 이야기이고, x번 집을 털지 않을 때 훔칠 수 있는 최대 액수는, "x - 1번집 까지 털었을 때의 최대액수"가 된다.
    - 위에서 소개한 변수로 표현해 본다면 DP[x] = DP[x - 1]이 된다는 것이다.

- 우리는 위에서 말한 2가지 경우(터는 경우, 털지 않는 경우) 를 비교해가면서 최대값을 구해보면 된다.
- 수식으로 나타내보면 DP[x] = max(DP[x - 2] + money[x] , DP[x - 1]) 이 되는 것이다.

```c++
#include <string>
#include <vector>

using namespace std;
int solution(vector<int> money) {
    int answer = 0;
    // 참고: https://yabmoons.tistory.com/477
 
    // 점화식: DP[x] = max(DP[x - 2] + money[x] , DP[x - 1]) 
    // DP1은 첫번째 집을 턴다는 가정, DP2는 첫번째 집을 털지 않는다는 가정
    vector<int> DP1(1000001,0), DP2(1000001,0);
    
    DP1[0] = money[0];
    DP1[1] = DP1[0];
    
    DP2[0] = 0;
    DP2[1] = money[1];
    
    for(int i = 2; i < money.size()-1; ++i)
        DP1[i] = max(DP1[i-2] + money[i] , DP1[i-1]);
    
    for(int i = 2; i < money.size(); ++i)
        DP2[i] = max(DP2[i-2] + money[i] , DP2[i-1]);
    
    return answer = max(DP1[money.size()-2], DP2[money.size()-1]);
}
```

## 끝