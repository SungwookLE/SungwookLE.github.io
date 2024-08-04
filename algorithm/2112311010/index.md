---
layout: post
type: algorithm
date: 2021-12-31 10:10
category: 코딩테스트 연습
title: 단어 변환 Lv3
subtitle: 프로그래머스 깊이/너비 우선 탐색(DFS/BFS)
writer: 100
hash-tag: [DFS, BFS, Programmers]
use_math: true
---

# 프로그래머스 > 깊이/너비 우선 탐색(DFS/BFS) > 단어 변환
> AUTHOR: SungwookLE    
> DATE: '21.12/31  
>> PROBLEM: [문제링크](https://programmers.co.kr/learn/courses/30/lessons/43163)  
>> LEVEL: Lv3    

## 1. 풀긴 풀었음 
- `words` 벡터에서 단어를 하나씩 체크하면서 한글자만 다른 단어인지 체크하고, 조건을 만족하면 해당 단어로 변환을 한다.
- 이런식으로 모든 경로에 대해 체크하고, 가능한 경로 중 최소 경로를 반환해 주는 방식으로 풀었다.

## 2. 코드
- 코드는 아래와 같다.

```c++
using namespace std;

class solver_this{
  public:
    solver_this(string _begin, string _target, vector<string> _words): begin(_begin), target(_target), words(_words){
        n = words.size();
    }
    bool checker(string now, string cmp){
        if (now != cmp){
            int same  = 0;
            for(int i = 0 ; i < now.length(); ++i)
                if ( now[i] == cmp[i])
                    same+=1;
            if (same == now.length()-1)
                return true;
        }
        return false;
    }
    
    void DFS(int iter, int answer, string now, vector<int> visited){
            for(int i = 0 ; i < words.size(); ++i){
                if (visited[i] == 0){
                    if (checker(now, words[i])){
                        string prev_now = now;
                        now = words[i];
                        visited[i] = 1;
                        cout << answer+1 << ": " << now << endl;
                        if (now == target){
                            cout << endl;
                            cases.push_back(answer+1);
                            return;
                        }
                        DFS(iter+1, answer+1, now, visited);
                        visited[i] = 0;
                        now = prev_now;
                    }
                }
            }
    }
    vector<int> cases;
    
  private:
    string begin, target;
    vector<string> words;
    int n;
    
};

int solution(string begin, string target, vector<string> words) {
    int answer = 0;
    
    solver_this solver(begin, target, words);
    
    int iter=0, mid_answer=0;
    string now =begin;
    vector<int> visited(words.size(),0);
        
    solver.DFS(iter, mid_answer, now, visited);
    
    if (solver.cases.size() ==0)
        answer = 0;
    else{
        sort(solver.cases.begin(), solver.cases.end());
        answer = *solver.cases.begin();
    }
    
    return answer;
}
```

## 끝