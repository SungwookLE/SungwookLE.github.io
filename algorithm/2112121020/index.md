---
layout: post
type: algorithm
date: 2021-12-12 10:20
category: 코딩테스트 연습
title: 네트워크 Lv3
subtitle: 프로그래머스 깊이/너비 우선 탐색(DFS/BFS)
writer: 100
hash-tag: [DFS, BFS, Programmers]
use_math: true
---


# 프로그래머스 > 깊이/너비 우선 탐색(DFS/BFS) > 네트워크  
> AUTHOR: SungwookLE    
> DATE: '21.12/12  
>> PROBLEM: [문제링크](https://programmers.co.kr/learn/courses/30/lessons/43162)  
>> LEVEL: Lv3    

## 1. 흠,, 두번째 시도하는 것인데,
- `DFS/BFS` 스타일은 이제 좀 느낌이 와서 시도를 해볼 수는 있었는데,
- 문제에 맞게 조건을 설정하는게 어려워서, 다른 사람의 코드를 쳐다봐야만 했다.. ㅠㅠ

## 2. 코드
- `if (from != i && computers[from][i] == 1 && visited[i] == 0 )` 이 조건문이 핵심이라고 할 수 있겠는데, 방문하지 않은 노드가 `from`가 이어져 있다면, `visited` 에 1을 넣어서, 전체 네트워크가 이어져있는지 아닌지 체크하게 된다.
- 한번의 루프를 돌았을 때, 여전히 방문하지 않은 노드가 존재한다면 두번째 네트워크가 존재한다는 의미가 된다.

```c++
//from to 'i'th computer
void bfs(int from,  vector<int>& visited,  vector<vector<int>> computers){
    for(int i = 0 ; i < computers.size(); ++i){
        if (from != i && computers[from][i] == 1 && visited[i] == 0 ){
            visited[i] = 1;
            bfs(i,  visited, computers);
        }
    }
}
int solution(int n, vector<vector<int>> computers) {
    int answer = 0;
    vector<int> visited(computers.size(), 0);
    
    for(int i = 0 ; i < computers.size(); ++i){
        if (visited[i] == 1){
            continue;
        }

        //아직도 방문이 되지 않은 노드가 남아있다고? ==> 그럼 네트워크는 하나 더 추가 되네!
        
        visited[i] = 1;
        answer +=1;
        
        bfs(i, visited, computers);
    }
    
    return answer;
}
```

## 끝