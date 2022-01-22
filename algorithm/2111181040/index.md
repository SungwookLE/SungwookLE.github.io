---
layout: post
type: algorithm
date: 2021-11-18 10:40
category: 코딩테스트 연습
title: 주식가격 Lv2
subtitle: 프로그래머스 스택/큐
writer: 100
hash-tag: [STACK, QUEUE, Programmers]
use_math: true
---



# 프로그래머스 > 스택/큐 > 주식가격
> AUTHOR: SungwookLE    
> DATE: '21.11/18  
>> PROBLEM: [문제링크](https://programmers.co.kr/learn/courses/30/lessons/42584)  
>> LEVEL: Lv2   

## 1. 풀이
- 쉬운 문제였음. 시간 복잡도는 O(1/2*N^2) 수준으로 봐야겠지

```c++
#include <string>
#include <vector>
using namespace std;

vector<int> solution(vector<int> prices) {
    vector<int> answer;
    vector<int> count;
    for(int i =0 ; i < prices.size() ; ++i){
        int cmp = prices[i];
        int tick = 0;
        for(int j = i+1 ; j <prices.size(); ++j){
            tick +=1;
            if ( cmp > prices[j])
                break;
        }
        count.push_back(tick);
    }
    
    return answer=count;
}
```

## 끝


