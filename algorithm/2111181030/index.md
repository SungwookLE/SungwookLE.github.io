---
layout: post
type: algorithm
date: 2021-11-18 10:30
category: 코딩테스트 연습
title: 다리를 지나는 트럭 Lv2
subtitle: 프로그래머스 스택/큐
writer: 100
hash-tag: [STACK, QUEUE, Programmers]
use_math: true
---



# 프로그래머스 > 스택/큐 > 다리를 지나는 트럭
> AUTHOR: SungwookLE    
> DATE: '21.11/18  
>> PROBLEM: [문제링크](https://programmers.co.kr/learn/courses/30/lessons/42583)  
>> LEVEL: Lv2   

## 1. 풀이
- `vector` 컨테이너에서 밀어내기 방식으로 문제 상황 그대로 구현하였다.
- 문제는 쉬운 편인데, 은근히 오래 걸리긴 한다. 코테 직전이라면 반복 풀이로 감을 끌어 올리긴 해야 한다. 시간 제한 안에 푸는게 시험이니깐

```c++
#include <string>
#include <vector>
#include <iostream>

using namespace std;

int solution(int bridge_length, int weight, vector<int> truck_weights) {
    int answer = 0;
    
    vector<int> on_bridge(bridge_length,0);
    int on_weights = 0;
    
    while(!truck_weights.empty()){
        int temp = *truck_weights.begin();
        on_weights -= on_bridge.back();
        on_weights += temp;

        if (on_weights <= weight){
            truck_weights.erase(truck_weights.begin());
            answer +=1;
            for(int i = on_bridge.size()-1 ; i > 0; --i)
                on_bridge[i] = on_bridge[i-1];
            on_bridge[0] = temp;
        }
        else{
            on_weights -= temp;
            answer +=1;
            for(int i = on_bridge.size()-1 ; i > 0; --i)
                on_bridge[i] = on_bridge[i-1];
            on_bridge[0] = 0;
        }
    }
    
    return answer+bridge_length;
}
```

## 끝


