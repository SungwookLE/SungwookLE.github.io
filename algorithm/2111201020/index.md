---
layout: post
type: algorithm
date: 2021-11-20 10:20
category: 코딩테스트 연습
title: 디스크 컨트롤러 Lv3
subtitle: 프로그래머스 힙
writer: 100
hash-tag: [HEAP, Programmers]
use_math: true
---

# 프로그래머스 > 힙 > 디스크 컨트롤러
> AUTHOR: SungwookLE    
> DATE: '21.11/20  
>> PROBLEM: [문제링크](https://programmers.co.kr/learn/courses/30/lessons/42627)  
>> LEVEL: Lv3    

## 1. 조금 오래 걸려서 풀긴 풀었음
- 조금 오래 걸려서 풀었는데, 의외로 풀어놓고 생각을 해보니, 매우 쉽게 푸는 문제이다.
- 그냥 이 정렬 규칙만 적용해서 `vector` 컨테이너를 정렬만 해서 앞에서 부터 꺼내서 출력해 내면 되는 것..

```c++
sort(jobs.begin(), jobs.end(), [last](auto a , auto b){
            if (last > a[0] && last > b[0]){
                if (a[1] < b[1])
                    return true;
                else if (a[1] == b[1]){
                    if (a[0] < b[0])
                        return true;
                    else
                        return false;
                }
                else
                    return false;
            } 
            else{
                if (a[0] < b[0])
                    return true;
                else if (a[0] == b[0]){
                    if (a[1] < b[1])
                        return true;
                    else
                        return false;
                }
                else
                    return false;
            }
        });
```
- 여기서 `last`는 실행 중인 `jobs`의 종료 예상 시간이다.
- `last`보다 시작시간이 빠른 그 다음의 `jobs`들은 실행 소모 시간이 짧은 순서로 정렬을 해야지, 전체 실행 소모 시간이 줄어든다.


## 2. 풀이

```c++
#include <string>
#include <vector>
#include <algorithm>

using namespace std;

int solution(vector<vector<int>> jobs) {
    int answer = 0;
    
    sort(jobs.begin(), jobs.end(), [](auto a, auto b){
        if (a[0] < b[0])
            return true;
        else if (a[0] == b[0]){
            if (a[1] < b[1])
                return true;
            else
                return false;
        }
        else
            return false;
    });
    
    auto temp = *jobs.begin();
    int last = temp[0];
    
    vector<int> ans;
    while(!jobs.empty()){
        ans.push_back(temp[1] + last - temp[0]);
        last = temp[1] + last;
        jobs.erase(jobs.begin());
        
        sort(jobs.begin(), jobs.end(), [last](auto a , auto b){
            if (last > a[0] && last > b[0]){
                if (a[1] < b[1])
                    return true;
                else if (a[1] == b[1]){
                    if (a[0] < b[0])
                        return true;
                    else
                        return false;
                }
                else
                    return false;
            } 
            else{
                if (a[0] < b[0])
                    return true;
                else if (a[0] == b[0]){
                    if (a[1] < b[1])
                        return true;
                    else
                        return false;
                }
                else
                    return false;
            }
        });
        temp = *jobs.begin();
    }
    
    int sum = 0;
    for(auto a : ans)
        sum+= a;
    return answer = sum / ans.size();
}
```
## 끝


