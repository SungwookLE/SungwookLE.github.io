---
layout: post
type: algorithm
date: 2021-11-28 10:10
category: 코딩테스트 연습
title: 조이스틱 Lv2
subtitle: 프로그래머스 그리디
writer: 100
hash-tag: [GREEDY, Programmers]
use_math: true
---

# 프로그래머스 > 그리디 > 조이스틱
> AUTHOR: SungwookLE    
> DATE: '21.11/28  
>> PROBLEM: [문제링크](https://programmers.co.kr/learn/courses/30/lessons/42860)  
>> LEVEL: Lv2    

## 1. 나의 풀이
- `그리디` 문제 방식으로 풀려고 생각을 해서 풀어봤는데,
- `forward` 순회만 해선 안되고, `backward` 순회 방식과 비교해서 낮은 값을 출력해주어야 했음
- 사람들이 올려둔 질문 보고 안되는 테스트 케이스 해결하니까 답이 풀렸음

```c++
int solution(string name) {
    int answer = 0;
    
    vector<int> cnts;    
    for(int i = 0 ; i < name.length(); ++i){
        cnts.push_back(name[i] - 'A');
        answer += min ( name[i]-'A', 'Z'- name[i]+1 );    
    }
    int idx = 0;
    
    int forward = 0;
    for(int i = 1 ; i < cnts.size() ; ++i){
        if(cnts[i] != 0){
            if ( (i-idx) <= cnts.size() - i){
                forward += i -idx;
                idx = i ;   
            }    
            else{
                forward += cnts.size()-i;
                idx = i;
            }
        }
    }
    
    idx = cnts.size()-1; 
    int backward = 1;
    for(int i = cnts.size()-1 ; i > 0 ; --i){
        if(cnts[i] != 0){
            if ( (idx-i) <= cnts.size() - i){
                backward += idx-i;
                idx = i ;   
            }    
            else{
                backward += cnts.size()-i;
                idx = i;
            }
        }
    }
    
    //cout << backward << endl;
    
    answer += min(forward,backward);
    return answer;
}
```

## 끝