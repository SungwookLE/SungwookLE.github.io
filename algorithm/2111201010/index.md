---
layout: post
type: algorithm
date: 2021-11-20 10:10
category: 코딩테스트 연습
title: 더 맵게 Lv2
subtitle: 프로그래머스 힙
writer: 100
hash-tag: [HEAP, Programmers]
use_math: true
---


# 프로그래머스 > 힙 > 더 맵게
> AUTHOR: SungwookLE    
> DATE: '21.11/20  
>> PROBLEM: [문제링크](https://programmers.co.kr/learn/courses/30/lessons/42626)  
>> LEVEL: Lv2   

## 1. 시간초과났었음
- `vector` 컨테이너를 이용해서 풀려고 하다 보니, 중간 중간에 오름차순으로 정렬하는 `sort`가 수행됬어야 했는데, 이것 때문에 시간 초과가 났었음
- `vector` 컨테이너가 아니고, `priority_queue`로 풀면 이를 해결할 수 있음

```c++
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>

using namespace std;

int solution(vector<int> scoville, int K) {
    int answer = 0;
    
    sort(scoville.begin(), scoville.end());
    
    
    bool flag = true;
    int new_scoville;
    
    while (flag){
        answer+=1;
        if (scoville.size() <= 1)
            return -1;
        
        new_scoville = scoville[0] + scoville[1]*2;
        scoville.erase(scoville.begin(), scoville.begin()+2);
        scoville.push_back(new_scoville);
        sort(scoville.begin(), scoville.end());
        
        if (scoville[0] >=K)
            flag = false;
    }

    return answer;
}
```

## 2. Priority_queue로 풀기
- 시간초과를 해결할 수 있었다.

```c++
#include <string>
#include <vector>
#include <queue>

using namespace std;

int solution(vector<int> scoville, int K) {
    int answer = 0;
    
    priority_queue<int, vector<int>, greater<int>> p_queue;
    for(int i = 0 ; i < scoville.size() ; ++i){
        p_queue.push(scoville[i]);
    }
 
    bool flag = true;
    int new_scoville;
    int fir,sec;
    
    while (flag){
        answer+=1;
        
        if (p_queue.size() <= 1)
            return -1;
        fir = p_queue.top(); p_queue.pop();
        sec = p_queue.top(); p_queue.pop();
        new_scoville =fir + sec*2;
        p_queue.push(new_scoville);
        
        if (p_queue.top() >=K)
            flag = false;
    }

    return answer;
}
```
## 3. priority_queue 정렬을 내 맘대로 하려면,
- STL 중 `priority_queue`가 유용할 때가 있을 듯하여 사용법 몇개를 정리해 본다.
- [참고 블로그](https://kbj96.tistory.com/15)를 살펴보자  
    - priority_queue의 정렬 방식을 커스터마이징하기
        -  일단은 `struct`를 만들어 주어야함
        
```c++
        struct Student {
            int id;
            int math, eng;
            Student(int num, int m, int e) : id(num), math(m), eng(e) {}    // 생성자 정의
        };
 
        // 학번을 기준으로 학번(id) 값이 큰 것이 Top 을 유지 하도록 한다.
        struct cmp {
            bool operator()(Student a, Student b) {
                return a.id < b.id;
            }
        };
 
        int main() {
            // 위에서 만든 cmp 구조체를 넣어 준다.
            priority_queue<Student, vector<Student>, cmp> pq;  
 
            // 이렇게 하면 priority_queue에 커스터마이징된 정렬이 되어 queue에 들어가게 된다.

            //...
            return 0;
        }
```

## 끝


