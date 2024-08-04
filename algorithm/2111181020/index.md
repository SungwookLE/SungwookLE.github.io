---
layout: post
type: algorithm
date: 2021-11-18 10:20
category: 코딩테스트 연습
title: 프린터 Lv2
subtitle: 프로그래머스 스택/큐
writer: 100
hash-tag: [STACK, QUEUE, Programmers]
use_math: true
---


# 프로그래머스 > 스택/큐 > 프린터
> AUTHOR: SungwookLE    
> DATE: '21.11/18  
>> PROBLEM: [문제링크](https://programmers.co.kr/learn/courses/30/lessons/42587)  
>> LEVEL: Lv2   

## 1. 풀이
- 문제를 읽고, 그대로 풀어야지만 빨리 푸는 것이다. 그래서 stack, queue (사실상 `vector`로 다 해결 가능)을 컨셉 그대로 쓰는게 중요하다.
- 처음에 문제를 잘못 해석해서, `train`이라는 vector에 담은 다음에 `sort`를 써서 풀었는데, 문제를 잘못 이해한 것으로 다시 풀었다.

```c++
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>

using namespace std;

int solution(vector<int> priorities, int location) {
    int answer = 0;
    
    vector<pair<int, int>> train, ans;
    for(int i =0 ; i < priorities.size() ; ++i){
        train.push_back(make_pair(i, priorities[i]));
    }
    
    while (!train.empty()){
        auto temp = *train.begin();
        train.erase(train.begin());

        bool checker = true;
        for(int i = 0 ; i < train.size() ; ++i){
            if (temp.second < train[i].second){
                train.push_back(temp);
                checker = false;
                break;
            }
        }
        if (checker){
            ans.push_back(make_pair(temp.first, temp.second));
        }
    }
    
    for(int i =0 ; i < ans.size() ; ++i){
        if ( ans[i].first == location){
            answer = i+1;
        }
    }
    return answer;
}
```

## 끝


