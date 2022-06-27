---
layout: post
type: algorithm
date: 2021-11-25 10:30
category: 코딩테스트 연습
title: 체육복 Lv1
subtitle: 프로그래머스 그리디
writer: 100
hash-tag: [GREEDY, Programmers]
use_math: true
---


# 프로그래머스 > 그리디 > 체육복
> AUTHOR: SungwookLE    
> DATE: '21.11/25  
>> PROBLEM: [문제링크](https://programmers.co.kr/learn/courses/30/lessons/42862)  
>> LEVEL: Lv1    

## 1. 나의 풀이
- 뭔가 상당히 복잡스럽게, 더럽게 코드를 만들었다. 통과는 하긴 했는데, 효율적인 방식이 생각이 안나네?
- 특히, `vector.erase`할 때 메모리 안나게 하려고, `sort`도 가져와서 썻어야만 했는데, 이렇게 푸는건 아닌듯 싶긴했다.
```c++
    sort(lost.begin(), lost.end());
    sort(reserve.begin(), reserve.end());
    
    vector<pair<int, int>> check;
    for(int i = 0 ; i < lost.size() ; ++i){
        for(int j =0 ; j < reserve.size() ; ++j){
            if (lost[i] == reserve[j]){
                check.push_back(make_pair(i,j));
                break;   
            }
        }
    }
    
    for(int idx = check.size()-1 ; idx >=0 ; --idx){
       lost.erase(lost.begin()+check[idx].first);
       reserve.erase(reserve.begin()+check[idx].second);
    }
```

- 내가 제출한 코드..  

```c++
#include <string>
#include <vector>
#include <algorithm>

using namespace std;
int solution(int n, vector<int> lost, vector<int> reserve) {
    int answer = 0;
    int restore=0;
    
    sort(lost.begin(), lost.end());
    sort(reserve.begin(), reserve.end());
    
    vector<pair<int, int>> check;
    for(int i = 0 ; i < lost.size() ; ++i){
        for(int j =0 ; j < reserve.size() ; ++j){
            if (lost[i] == reserve[j]){
                check.push_back(make_pair(i,j));
                break;   
            }
        }
    }
    
    for(int idx = check.size()-1 ; idx >=0 ; --idx){
       lost.erase(lost.begin()+check[idx].first);
       reserve.erase(reserve.begin()+check[idx].second);
    }
    
    for(int i = 0 ; i < lost.size() ; ++i){
        for(int j =0 ; j < reserve.size() ; ++j){
            if(lost[i]-1 == reserve[j] || lost[i]+1 == reserve[j]){
                restore +=1;
                reserve.erase(reserve.begin()+j);
                break;
            }   
        }
    }
    
    answer = n-lost.size() + restore ;
    return answer;
}
```

## 2. 다른사람 풀이
- 확실히 군더더기 없이 깔끔하다.
- 배열에다가 체육복의 개수 정보를 넣어두고,
- `lost`와 `reserve`가 같을 때 `+1, -1`로 처리를 해주었다..

```c++
#include <string>
#include <vector>

using namespace std;
int student[35];
int solution(int n, vector<int> lost, vector<int> reserve) {
    int answer = 0;
    for(int i : reserve) student[i] += 1;
    for(int i : lost) student[i] += -1;
    for(int i = 1; i <= n; i++) {
        if(student[i] == -1) {
            if(student[i-1] == 1) 
                student[i-1] = student[i] = 0;
            else if(student[i+1] == 1) 
                student[i] = student[i+1] = 0;
        }
    }
    for(int i  = 1; i <=n; i++)
        if(student[i] != -1) answer++;

    return answer;
}
```

## 끝