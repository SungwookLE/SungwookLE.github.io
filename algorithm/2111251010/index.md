---
layout: post
type: algorithm
date: 2021-11-25 10:10
category: 코딩테스트 연습
title: 소수찾기 Lv2
subtitle: 프로그래머스 완전탐색
writer: 100
hash-tag: [FULL_SEARCH, Programmers]
use_math: true
---

# 프로그래머스 > 완전탐색 > 소수찾기 
> AUTHOR: SungwookLE    
> DATE: '21.11/25  
>> PROBLEM: [문제링크](https://programmers.co.kr/learn/courses/30/lessons/42839)  
>> LEVEL: Lv2    

## 1. 풀이
- FULL_SEARCH를 하기 위해서, 백트래킹을 활용하여 숫자 조합을 만드는 함수를 만들었고 (`void make_combi`), 함수를 통해 나온 숫자 조합을 가지고, 해당 숫자가 소수인지 아닌지 세어보았다.
- `vector` 컨테이너에서 중복된 원소를 제거하기 위한 방법은 자주 쓰이니까, 할 때마다 검색하지 말고 알아두자
```c++
sort(v.begin(), v.end());
v.erase(unique(v.begin(), v.end()), v.end());
```
- 백트래킹 방식은 문제 푸는데 있어, 유용한 해결책이 되는 경우가 꽤 있는 듯 하다.

```c++
#include <string>
#include <vector>
#include <algorithm>

bool isprime(int num){
    
    if(num == 0 || num == 1)
        return false;
    
    for(int i = 2; i < (num/2+1) ; ++i){
        if(num % i == 0)
            return false;
    }
    return true;
}

void make_combi(int iter, int n, string numbers, vector<int> check, string comb, vector<int>& combs){
    if (iter == n){
        comb ="";
        return;
    }
    else{
        
        for(int i = 0 ; i < numbers.length(); ++i){
            if (check[i] == 0){
                check[i] = 1;
                comb += numbers[i];
                combs.push_back(stoi(comb));
                make_combi(iter+1, n, numbers, check, comb, combs);
                comb.erase(comb.length()-1, 1);
                check[i] = 0;
            }
        }
    }
}

int solution(string numbers) {
    int answer = 0;
    int iter =0;
    int n = numbers.length();
    vector<int> check(n,0);
    string comb ="";
    vector<int> combs;
    
    make_combi(iter, n, numbers, check, comb, combs);
    
    sort(combs.begin(), combs.end());
    combs.erase(unique(combs.begin(), combs.end()), combs.end());

    for(auto num : combs){
        if (isprime(num))
            answer+=1;
    }
    
    return answer;
}
```

## 끝