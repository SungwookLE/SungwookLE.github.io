---
layout: post
type: algorithm
date: 2021-11-23 10:10
category: 코딩테스트 연습
title: 모의고사 Lv1
subtitle: 프로그래머스 완전탐색
writer: 100
hash-tag: [FULL_SEARCH, Programmers]
use_math: true
---


# 프로그래머스 > 완전탐색 > 모의고사 
> AUTHOR: SungwookLE    
> DATE: '21.11/23  
>> PROBLEM: [문제링크](https://programmers.co.kr/learn/courses/30/lessons/42840)  
>> LEVEL: Lv1    

## 1. 풀이
- 오늘 머리가 이상하게 굳어서, 정렬하는 것에 있어서 아이디어가 잘 안 떠올랐다.. 쩝;
- 문제는 간단한 것이다. 마지막에 정답 출력하는 쪽에서 이상하게 머리가 안 굴러갔다.

```c++
using namespace std;
vector<int> solution(vector<int> answers) {
    vector<int> answer;
    
    vector<int> a = {1,2,3,4,5};
    vector<int> b = {2,1,2,3,2,4,2,5};
    vector<int> c = {3,3,1,1,2,2,4,4,5,5};
    
    int idx = 0;
    int score_a =0, score_b=0, score_c=0;
    for(int ans : answers){
        if (ans == a[idx%a.size()])
            score_a +=1;
        if (ans == b[idx%b.size()])
            score_b +=1;
        if (ans == c[idx%c.size()])
            score_c +=1;
        idx+=1;
    }
    
    //사실상 `vector<pair<int,int>>`도 쓸 필요 없음
    vector<pair<int, int>> scores;
    scores.push_back(make_pair(1, score_a));
    scores.push_back(make_pair(2, score_b));
    scores.push_back(make_pair(3, score_c));
    
    int max =0;
    for(int i = 0 ; i < scores.size() ; ++i){
        if (max < scores[i].second)
            max = scores[i].second;
    }
    
    for(int i = 0 ; i < scores.size() ; ++i){
        if (max == scores[i].second)
            answer.push_back(i+1);
    } 
    return answer;
}
```

## 끝