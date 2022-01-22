---
layout: post
type: algorithm
date: 2021-11-18 10:10
category: 코딩테스트 연습
title: 기능개발 Lv2
subtitle: 프로그래머스 스택/큐
writer: 100
hash-tag: [STACK, QUEUE, Programmers]
use_math: true
---


# 프로그래머스 > 스택/큐 > 기능개발
> AUTHOR: SungwookLE    
> DATE: '21.11/18  
>> PROBLEM: [문제링크](https://programmers.co.kr/learn/courses/30/lessons/42586)  
>> LEVEL: Lv2   

## 1. 풀이
- 사실상 `vector`라는 컨테이너를 잘 활용하는지 체크하는 문제이다.

```c++
#include <string>
#include <vector>
#include <iostream>

using namespace std;

vector<int> solution(vector<int> progresses, vector<int> speeds) {
    vector<int> answer;
    
    vector<double> v(progresses.size(), 100);
    for(int i = 0 ; i < progresses.size() ; ++i)
        v[i] = (v[i] - double(progresses[i]))/double(speeds[i]);
    
    while(!v.empty()){
        for(int i = 0 ; i < v.size() ; ++i){
            v[i] -= 1;
            //cout << v[i] << " ";
        }
        //cout << endl;
        
        int count =0 ;
        while(v[0] <= 0 && !v.empty()){
            count +=1;
            v.erase(v.begin());
        }
        
        if (count > 0){
            answer.push_back(count);
        }
    }
    
    return answer;
}
```

## 끝


