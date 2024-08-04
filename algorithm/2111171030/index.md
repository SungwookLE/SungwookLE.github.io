---
layout: post
type: algorithm
date: 2021-11-17 10:30
category: 코딩테스트 연습
title: 위장 Lv2
subtitle: 프로그래머스 HASH
writer: 100
hash-tag: [HASH, Programmers]
use_math: true
---



# 프로그래머스 > 해쉬 > 위장
> AUTHOR: SungwookLE    
> DATE: '21.11/17  
>> PROBLEM: [문제링크](https://programmers.co.kr/learn/courses/30/lessons/42578)  
>> LEVEL: Lv2  

## 1. 풀이
- `unordered_map` 같은 컨테이너를 활용하면 `dict` 데이터 정리에 유용하다.
- 의상 종류 별로 개수를 센다음에 곱하는데, 이 때 압입는 경우도 생각해서 +1 한 것을 곱한다.
- 마지막에, 아예 안 입는 것은 안되니까 1을 빼준다.

```c++
#include <string>
#include <vector>
#include <algorithm>
#include <unordered_map>
using namespace std;

int solution(vector<vector<string>> clothes) {
    int answer = 1;
    unordered_map<string, int> map;
    
    for(int i =0 ; i < clothes.size() ; ++i){
        string kind = clothes[i].back();
        map[kind] +=1;
    }
    
    for(auto it = map.begin(); it != map.end() ; ++it){
        answer *= (it->second+1);
        // 안입는 경우를 더해서 다 곱한다음에, 아무것도 안입는 경우의 수는 없으니까 1을 뺸다.. 
    }
    answer -= 1;
     
    return answer;
}
```

## 끝


