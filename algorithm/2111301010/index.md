---
layout: post
type: algorithm
date: 2021-11-30 10:10
category: 코딩테스트 연습
title: 큰 수 만들기 Lv2
subtitle: 프로그래머스 그리디
writer: 100
hash-tag: [GREEDY, Programmers]
use_math: true
---


# 프로그래머스 > 그리디 > 큰 수 만들기
> AUTHOR: SungwookLE    
> DATE: '21.11/30  
>> PROBLEM: [문제링크](https://programmers.co.kr/learn/courses/30/lessons/42883)  
>> LEVEL: Lv2    

## 1. 문제 풀이
- `그리디` 문제는 순회 문제에서, full-search를 하지 않아도 중간 중간 `sub-optimal` Law를 통해 답을 찾는 방식을 말한다.
- 아이디어가 잘 떠올라야하는 문제 유형이고, 잘 떠오르지 않았던 문제이다.
- **IDEA:** number를 돌면서 i번째 값이 i+1번째 값보다 작으면 지워주고 break하는 것을 k번 반복한다.

```c++
string solution(string number, int k) {
    string answer = "";
    while(k > 0){
        bool checker = false;
        for(int i =0 ; i < number.length()-1 ; ++i){
            if ( number[i] < number[i+1]){
                number.erase(i, 1);
                checker = true;
                break;
            }
        }
        if (checker==false)
            number.erase(number.length()-1,1);
        k--;
    }
    return answer = number;
}
```

## 끝