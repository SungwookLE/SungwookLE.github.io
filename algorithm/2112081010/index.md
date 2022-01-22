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

# 프로그래머스 > 동적계획법 > 등굣길
> AUTHOR: SungwookLE    
> DATE: '21.12/08  
>> PROBLEM: [문제링크](https://programmers.co.kr/learn/courses/30/lessons/42898#)  
>> REFERENCE: [참고](https://ongveloper.tistory.com/84)  
>> LEVEL: Lv3    

## 1. 혼자서 푸는 중.. 실패
- 이게 좀 `A-star` 알고리즘 응용하면 되겠다는 생각을 잡아서,, 트라이 해보았는데, (12/09 오전12:13분) 틀림..
- 하,, dynamic programing은 `memoization` 아이디어 잡는 것이 핵심이라는데 아이디어 잡기가 어렵다. -> **점화식**
- `DP`는 큰 문제를 작은 문제로 쪼개고, 작은문제의 답이 큰 문제의 중간답이 되게끔 코드를 짜는 것이다....

```c++
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

## 2. 다른 사람의 코드 참고
- 그리드 하나하나 씩 돌면서, 이전 그리드에 도달할 수 있는 경로의 개수를 더 해나가는 방식으로 풀었다.

```c++
if (DP[row-1][col] != -1)
  a = DP[row-1][col];
if (DP[row][col-1] != -1)
  b = DP[row][col-1];

DP[row][col] += (a+b)%1000000007; // 이전 그리드에 도달할 수 있는 경로의 개수를 더 해나가는 방식
```

- 하.. 한문제 한문제가 참 어렵고만 ㅋㅋ, 참고의 링크가 잘 정리가 되어 있다. 참고하자.


```c++
#include <string>
#include <vector>
#include <iostream>

using namespace std;

int solution(int m, int n, vector<vector<int>> puddles) {
    int answer = 0;
    
    vector<vector<int>> DP(101, vector<int> (101,0));
    DP[1][1] = 1;
    
    for(auto p : puddles){
        DP[p[1]][p[0]] = -1;
    }
    
    for(int row =1; row <= n ; ++row){
        for(int col=1; col <= m ; ++col){
            int a=0, b=0;
            if (DP[row][col] == -1)
                continue;
            if (DP[row-1][col] != -1)
                a = DP[row-1][col];
            if (DP[row][col-1] != -1)
                b = DP[row][col-1];
            
            DP[row][col] += (a+b)%1000000007;
        }
    }

    return answer = DP[n][m];
}
```

## 끝