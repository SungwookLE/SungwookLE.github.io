---
layout: post
type: algorithm
date: 2021-11-20 10:30
category: 코딩테스트 연습
title: 이중우선순위 큐 Lv3
subtitle: 프로그래머스 힙
writer: 100
hash-tag: [HEAP, Programmers]
use_math: true
---


# 프로그래머스 > 힙 > 이중우선순위 큐
> AUTHOR: SungwookLE    
> DATE: '21.11/20  
>> PROBLEM: [문제링크](https://programmers.co.kr/learn/courses/30/lessons/42628)  
>> LEVEL: Lv3    

## 1. 풀이
- `stringstream` 객체를 이용하여서, string을 split 하였음
- `priority_queue`를 사용하여 조금 더 알고리즘 효율적으로 풀려고 햇는데, 보니까, `priority_queue`는 한번 정렬을 해서 트리에 넣어버리고 꺼내쓰는 형태라서 `iterator`를 제공을 안했다.
- 이게, 안타까웠던 게, 이중의 정렬 규칙을 가지고 있어야 하는 문제였기 때문에 `v.end()` 이런식으로 데이터를 꺼내서 보고, 삭제하려고 햇는데 이게 안되어서, `vector`를 이용해서 데이터를 담고, `sort`로 정렬하였다.
- 이렇게 한 이유는, `vector`는 `iterator`를 제공하므로, `begin(), end()`등을 이용해서 데이터를 꺼내고 삭제할 수 있기 때문이었다.
- [stackoverflow 발췌:](https://stackoverflow.com/questions/4484767/how-to-iterate-over-a-priority-queue) `priority_queue` doesn't allow iteration through all the members, presumably because it would be too easy in invalidate the priority ordering of the queue (by modifying the elements you traverse) or maybe it's a "not my job" rationale.

```c++
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <algorithm>

using namespace std;

vector<int> solution(vector<string> operations) {
    vector<int> answer;

    vector<int> v;
    for(int i = 0 ; i < operations.size(); ++i){
        string temp = operations[i];
        stringstream ss(temp);
        string key, val;
        ss >> key >> val;
        
        if (key == "I"){
            v.push_back(stoi(val));
            sort(v.begin(), v.end());
        }
        else{
            if (!v.empty()){
                if (val == "1")
                    v.pop_back();
                else if (val == "-1")
                    v.erase(v.begin());
            }
        }
    }
    
    if (!v.empty())
        answer = {v.back(), *v.begin()};
    else
        answer = {0,0};
    return answer;
}
```


## 끝


