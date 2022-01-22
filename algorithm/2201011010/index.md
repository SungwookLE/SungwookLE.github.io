---
layout: post
type: algorithm
date: 2022-01-01 10:10
category: 코딩테스트 연습
title: 여행경로 Lv3
subtitle: 프로그래머스 깊이/너비 우선 탐색(DFS/BFS)
writer: 100
hash-tag: [DFS, BFS, Programmers]
use_math: true
---


# 프로그래머스 > 깊이/너비 우선 탐색(DFS/BFS) > 여행경로
> AUTHOR: SungwookLE    
> DATE: '22.01/01  
>> PROBLEM: [문제링크](https://programmers.co.kr/learn/courses/30/lessons/43164)  
>> LEVEL: Lv3    

## 1. 풀긴 풀었음
- 모든 티켓이 다 사용 가능할 떄, 들리게 되는 공항들을 `path`라는 벡터에 담았다. 
- 모든 `path`에 대한 경우의 수들을 최종적으로 `ports`라는 2차원 벡터에 담아 구하였다.
- 구한 `ports`에서 알파벳 순서로 정렬하여 최종 답을 구하였다.

## 2. 코드
- 코드는 아래와 같다.

```c++
class solver_this{
  public:
    solver_this(vector<vector<string>> _tickets): tickets(_tickets) {}
    bool used_all(vector<int> used_ticket){
        for (auto u : used_ticket){
            if (u != 1)
                return false;
        }
        return true;
    }
    
    void bfs(vector<int> used_ticket, vector<string> path){
        if (used_all(used_ticket)){
            ports.push_back(path);
            return;
        }
        else{
            for(int i =0 ; i < tickets.size() ; ++i){
                if (used_ticket[i] == 0){
                    if (path.back() == tickets[i][0]){
                        used_ticket[i] = 1;
                        path.push_back(tickets[i][1]);
                        bfs(used_ticket, path);
                        
                        used_ticket[i] = 0;
                        path.pop_back();
                    }                    
                }
            }
        }
    }
    
    vector<string> picking_path(){
        sort(ports.begin(), ports.end());
        return *ports.begin();
    }
    
    private:
    vector<vector<string>> tickets;
    vector<vector<string>> ports;
};

vector<string> solution(vector<vector<string>> tickets) {
    vector<string> answer;
    
    solver_this solver(tickets);
    vector<int> used_ticket(tickets.size(), 0);
    vector<string> path = {"ICN"};
        
    solver.bfs(used_ticket, path);
    answer = solver.picking_path(); 
   
    return answer;
}
```

## 끝