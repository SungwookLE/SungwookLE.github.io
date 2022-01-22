---
layout: post
type: algorithm
date: 2021-12-12 10:10
category: 코딩테스트 연습
title: 타겟 넘버 Lv2
subtitle: 프로그래머스 깊이/너비 우선 탐색(DFS/BFS)
writer: 100
hash-tag: [DFS, BFS, Programmers]
use_math: true
---


# 프로그래머스 > 깊이/너비 우선 탐색(DFS/BFS) > 타겟 넘버
> AUTHOR: SungwookLE    
> DATE: '21.12/12  
>> PROBLEM: [문제링크](https://programmers.co.kr/learn/courses/30/lessons/43165)  
>> LEVEL: Lv2    

## 1. 예전에 한번 풀었었음..
- 한번 풀었던 문제이기도 하고, 풀이가 기억나는 게 있어서 풀기는 함
- 숫자를 더하고 빼는 두가지 경우에 대해서 모든 경우를 다 계산하다가, 마지막에 `sum`과 `target`이 맞는지 비교하여 답 출력
- 알고리즘은 `Brute Force` 처럼 모든 경우를 탐색하는 것과 동일하다. 

## 2. 코드
- 코드  

```c++
#include <string>
#include <vector>

void bfs(int iter, int n, int sum, int target, vector<int> number, int& answer){
    if (iter == n){
        if (sum == target){
            answer +=1;
        }
    }
    else{
        //더하는 케이스 1)
        sum += number[iter];
        bfs(iter+1, n, sum, target, number, answer);
        //원상복귀
        sum -= number[iter];

        //빼는 케이스 2)
        sum -= number[iter];
        
        bfs(iter+1, n, sum, target, number, answer);
    }
}

int solution(vector<int> numbers, int target) {
    int answer = 0;
    
    int iter = 0, n = numbers.size();
    int sum = 0;
    
    bfs(iter, n, sum, target, numbers, answer);
    
    return answer;
}
```

## 끝