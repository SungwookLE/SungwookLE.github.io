---
layout: post
type: algorithm
date: 2021-11-25 10:20
category: 코딩테스트 연습
title: 카펫 Lv2
subtitle: 프로그래머스 완전탐색
writer: 100
hash-tag: [FULL_SEARCH, Programmers]
use_math: true
---


# 프로그래머스 > 완전탐색 > 카펫
> AUTHOR: SungwookLE    
> DATE: '21.11/25  
>> PROBLEM: [문제링크](https://programmers.co.kr/learn/courses/30/lessons/42842)  
>> LEVEL: Lv2    

## 1. 풀이
- 그냥, 노가다로 푸는 문제임, 쉬운 문제

```c++
using namespace std;
vector<int> solution(int brown, int yellow) {
    vector<int> answer;
    
    int yellow_h, yellow_w;
    vector<pair<int, int>> yellow_comb;
    for(int i = 1 ; i <= yellow/2+1; ++i){
        yellow_w = i;
        if (yellow%yellow_w ==0){
            yellow_h = yellow/yellow_w;
            if (yellow_w >= yellow_h)
                yellow_comb.push_back(make_pair(yellow_w, yellow_h));
        }
    }

    int brown_count;
    for(auto yy : yellow_comb){
        brown_count =0;
        brown_count += (yy.first)*2;
        brown_count += (yy.second)*2;
        brown_count +=4;
        
        if (brown_count == brown)
            return {yy.first+2, yy.second+2};
    }
    
    return answer;
}
```

## 끝