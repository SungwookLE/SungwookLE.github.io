---
layout: post
type: algorithm
date: 2021-12-08 10:10
category: 코딩테스트 연습
title: 등굣길 Lv3
subtitle: 프로그래머스 동적계획법
writer: 100
hash-tag: [Dynamic_Programming, Programmers]
use_math: true
---

- toc
{:toc}

# 프로그래머스 > 동적계획법 > 등굣길
> AUTHOR: SungwookLE    
> DATE: '21.12/08  
>> PROBLEM: [문제링크](https://programmers.co.kr/learn/courses/30/lessons/42898#)  
>> REFERENCE: [참고](-)  
>> LEVEL: Lv3    

## 1. 혼자서 푸는 중..
- 이게 좀 변칙으로 생각을 해서 코드 실행해봤는데,, (12/09 오전12:13분) 틀린다고 나오네,,,
- 내일 또 고민을 해보자..

```c++
#include <string>
#include <vector>

using namespace std;

int solution(int m, int n, vector<vector<int>> puddles) {
    int answer = 1;
    
    
    vector<vector<int>> opens;
    vector<vector<int>> next;
    vector<vector<int>> delta = { {1,0} , {0,1}}; // down, right
    
    
    opens.push_back({0,0});
    vector<int> goal = {n-1, m-1};
    int count = 0;
    
    while(opens.back()[0] != goal[0] || opens.back()[1] != goal[1]){
        
        count +=1;    
        for(auto a : opens){
            int count = 0;
            
            for(auto d : delta){
                int row2 = a[0] + d[0];
                int col2 = a[1] + d[1];
                
                if (row2 >= 0 && row2 < n && col2 >= 0 && col2 < m){
                    bool is_puddles = false;
                    for(auto p : puddles){
                        if (row2 == p[0]-1 && col2 ==p[1]-1){
                            is_puddles =true;
                            break;
                        }
                    }

                    if (is_puddles==false){
                        count +=1;
                        next.push_back({row2,col2});
                    }
                }
            }

            if (count == 2){
                answer+=1;
            }
        }
        opens.clear();
        opens= next;
        next.clear();
    }
    
    return answer;
}
```
## 끝(아직)
