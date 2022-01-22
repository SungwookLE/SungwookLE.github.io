---
layout: post
type: algorithm
date: 2022-01-02 10:20
category: 코딩테스트 연습
title: 징검다리 Lv4
subtitle: 프로그래머스 이분탐색
writer: 100
hash-tag: [Binary_Search, Max_index_Find, Programmers]
use_math: true
---

# 프로그래머스 > 이분탐색 > 징검다리
> AUTHOR: SungwookLE    
> DATE: '22.01/02  
>> PROBLEM: [문제링크](https://programmers.co.kr/learn/courses/30/lessons/43236)  
>> LEVEL: Lv4    

## 1. 문제 풀이 (혼자 못 풀었음)
- 문제를 어떻게 이분탐색으로 풀 수 있을지 생각이 안 떠올랐다.
- 이분탐색을 사용해서 푸는 `vector` 요소들 중에 제일 적합한 요소(방정식의 해라던지)를 피킹하는 알고리즘은 생각이 나는데,
- 이런 일반적인 문제를 푸는 방법으로 `이분탐색`이 가능하겠다.

- 풀이
    1. 최대거리: 0 부터 끝까지 (~distance)
    2. 최소거리: 1
    3. 최적거리: 최대거리와 최소거리 사이의 어딘가.. (mid)

    4. 내가 고른 mid가 offset이 되었을 때, `offset >= mid` 조건을 만족하는 `rocks`의 개수를 센다.
    5. 조건을 만족하는 `rocks`가 `전체바위-제거한바위` 개수보다 크다면 `end = mid-1`, 작으면 `start = mid+1` 을 하여 중간 값을 찾아 나간다.

## 2. 코드
- Binary Search 알고리즘에 조금의 변형이 들어간 것으로 조건을 만족하는 Max_index 를 찾는 알고리즘이다.

```c++
#include <string>
#include <vector>
#include <algorithm>

using namespace std;

class solver_this{
  public:
    solver_this(int _distance, vector<int> _rocks, int _n): distance(_distance), rocks(_rocks), n(_n){
        sort(rocks.begin(), rocks.end());
        total_rocks = rocks.size();
    }
    
    void solver(){
        
        int start = 1;
        int end = distance;
        int mid;
        
        while (start <= end){
            mid = (start+end)/2;
            int last_offset = 0;
            int rock_cnt =0;
            
            for (int i = 0 ; i < total_rocks -1 ; ++i){
                int offset = rocks[i] - last_offset;
                if (offset >= mid){
                    last_offset = rocks[i];
                    rock_cnt +=1;
                }
            }
            
            if (rocks.back() - last_offset >= mid &&
               distance - rocks.back() >= mid)
                rock_cnt +=1;
            
            if (rock_cnt >= total_rocks -n)
                start = mid + 1;
            else
                end = mid-1;
        }
        answer = end;
    }
    
    int get_answer(){
        return answer;
    }
    
  private:
    int distance, n;
    vector<int> rocks;
    int total_rocks;
    int answer;
};

int solution(int distance, vector<int> rocks, int n) {
    int answer = 0;
    
    solver_this solver(distance, rocks, n);
    solver.solver();
    answer = solver.get_answer();
    
    return answer;
}
```

## 끝