---
layout: post
type: algorithm
date: 2021-11-17 10:20
category: 코딩테스트 연습
title: 전화번호 목록 Lv2
subtitle: 프로그래머스 HASH
writer: 100
hash-tag: [HASH, Programmers]
use_math: true
---



# 프로그래머스 > 해쉬 > 전화번호 목록
> AUTHOR: SungwookLE    
> DATE: '21.11/17  
>> PROBLEM: [문제링크](https://programmers.co.kr/learn/courses/30/lessons/42577)  
>> LEVEL: Lv2  

## 1. 비효율적인 방법
- 이것도 굳이 for 문을 2번 돌릴 필요가 없다. 아래의 코드는 for문을 2번 썼기에 O(N2)이고 시간초과가 된다.
- `sort`를 하게되면 알파벳 순서로 정렬이 되기 때문이다.

```c++
bool solution(vector<string> phone_book) {
    bool answer = true;

    sort(phone_book.begin(), phone_book.end());
    
    for(int i=0; i<phone_book.size(); ++i){
        int leng = phone_book[i].size();
        for(int j=(i+1); j<phone_book.size(); ++j){
            string front = phone_book[j].substr(0, leng);
            if (front == phone_book[i])
                return false;
            
        }        
    }
    
    return answer;
}
```

## 2. 효율적인 방법
```c++
bool solution(vector<string> phone_book) {
    bool answer = true;

    sort(phone_book.begin(), phone_book.end());
   
    for(int i=0; i<phone_book.size()-1; ++i){
        int leng = phone_book[i].size();
        string front = phone_book[i+1].substr(0, leng);
        if (front == phone_book[i])
               return false;
    }
    
    return answer;
}
```

## 끝


