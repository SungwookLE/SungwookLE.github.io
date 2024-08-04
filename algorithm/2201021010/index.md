---
layout: post
type: algorithm
date: 2022-01-02 10:10
category: 코딩테스트 연습
title: 입국심사 Lv3
subtitle: 프로그래머스 이분탐색
writer: 100
hash-tag: [Binary_Search, Min_index_Find, Programmers]
use_math: true
---


# 프로그래머스 > 이분탐색 > 입국심사
> AUTHOR: SungwookLE    
> DATE: '22.01/02  
>> PROBLEM: [문제링크](https://programmers.co.kr/learn/courses/30/lessons/43238)  
>> LEVEL: Lv3    

## 1. 문제 풀이
- 어떻게 이 문제가 이분탐색로 풀 수 있는지 생각 나지 않아, 단순하게 `timer`를 하나씩 증가시키고, `timer`와 `times` 요소의 나머지가 0인 지점에서 사람을 한명씩 제껴나가는 방식으로 코드를 구성하였으나, 당연하게도 `시간초과`
- 이분탐색으로 푸는 풀이 해설들을 보면서 코드를 작성하였다.
    - 최소시간: 한 명의 입국자가 최소 시간의 입국 심사관에게 심사받는 시간
    - 최대시간: 모든 입국자들이 최소 시간의 입국 심사관에게 심사받는 시간
    - 최적 시간: 최소시간과 최대시간 사이에 위치할 것이다.
    - 이 사이에 위치한 시간을 이분탐색으로 찾는 문제가 되겠다.

## 2. 코드
- 이분탐색 코드를 사용하는 것인데, 이 문제는 이분탐색의 조금 변형이 가미된 것으로 조건을 만족하는 최소값을 찾는 문제가 된다.

```c++
#include <string>
#include <vector>
#include <algorithm>

using namespace std;

class solver_this{
    public:
    solver_this(int _n, vector<int> _times): n(_n), times(_times) {sort(times.begin(), times.end());
                                                                  longest_time = times[0] * n;}
    void find_fastest_time(){
        long long start = times[0];
        long long end = longest_time;
        long long mid;
        
        while( start <= end){
            mid = (start+end)/2;
            int checked_person=0;
            
            for(int i =0 ; i < times.size(); ++i){
                checked_person += mid / times[i];
                if (checked_person >= n){
                    end = mid-1;
                    break;
                }
            }
       
            if (checked_person < n)
                start = mid+1;
        }
        
        answer = start;
    }
    
    long long get_answer(){
        return answer;
    }
    
    private:
    int n;
    vector<int> times;
    long long longest_time;  
    long long answer;
    
};

long long solution(int n, vector<int> times) {
    long long answer = 0;
    
    solver_this solver(n, times);
    solver.find_fastest_time();
    answer = solver.get_answer();
    
    return answer;
}
```

## 끝