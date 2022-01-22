---
layout: post
type: algorithm
date: 2021-11-17 10:40
category: 코딩테스트 연습
title: 베스트앨범 Lv3
subtitle: 프로그래머스 HASH
writer: 100
hash-tag: [HASH, Programmers]
use_math: true
---



# 프로그래머스 > 해쉬 > 베스트앨범
> AUTHOR: SungwookLE    
> DATE: '21.11/17  
>> PROBLEM: [문제링크](https://programmers.co.kr/learn/courses/30/lessons/42579)  
>> LEVEL: Lv3  

## 1. 풀이
- `unordered_map` 같은 컨테이너를 활용하면 `dict` 데이터 정리에 유용한데, 이번 문제에는 중복 key 카운팅할 때 매우 편리하였다. 
```c++
unordered_map<string, int> genres_count;
for(int i =0 ; i < genres.size() ; ++i){
    genres_count[genres[i]]+=plays[i];
}
```
- map 컨테이너를 가지고 custom의 정렬을 수행하는게 좀 까다로웠다. 검색을 거쳐서 결국엔 vector 컨테이너로 데이터를 옮겨와서 `sort`하였다. 
- `copy`라인을 보면 unordered_map 데이터를 vector 컨테이너로 들고 왔음을 알 수 있다.

```c++
#include <string>
#include <vector>
#include <algorithm>
#include <unordered_map>

using namespace std;

vector<int> solution(vector<string> genres, vector<int> plays) {
    vector<int> answer;
    unordered_map<string, int> genres_count;
    
    for(int i =0 ; i < genres.size() ; ++i){
        genres_count[genres[i]]+=plays[i];
    }
    
    
    unordered_map<int, vector<int>> feature;
    for(int i =0 ; i < genres.size() ; ++i){
        feature[i] = {genres_count[genres[i]] , plays[i]};
    }
    
    
    vector< pair<int, vector<int> >> v;
    copy(feature.begin(), feature.end(), back_inserter<vector<pair<int, vector<int> >>>(v));
    
    sort(v.begin(), v.end(), [](auto a, auto b){
        if (a.second[0] > b.second[0])
            return true;
        else if ( a.second[0] == b.second[0]){
            if ( a.second[1] > b.second[1])
                return true;
            else if ( a.second[1] == b.second[1]){
                if ( a.first < b.first)
                    return true;
                else
                    return false;
            }
            else
                return false;
        }
        else
            return false;
    });
    
    unordered_map<int, int> counter;
    for(auto it = v.begin() ; it != v.end() ; ++it){
        
        counter[it->second[0]]+=1;
        if ( counter[it->second[0]] <= 2)
            answer.push_back(it->first);
        
    }
    return answer;
}
```

## 끝


